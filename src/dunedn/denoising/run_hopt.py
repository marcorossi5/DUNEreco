from hyperopt import fmin, tpe, hp, Trials, space_eval, STATUS_OK
from hyperopt.mongoexp import MongoTrials
from time import time as tm
import argparse, os, shutil, sys, datetime, yaml, pprint, pickle
import dunedn.denoising.denoise as denoise
import dunedn.denoising.analysis as analysis
from dunedn.denoising.args import Args


def run_hyperparameter_scan(search_space, max_evals, cluster, folder):
    """Running hyperparameter scan using hyperopt"""
    print("[+] Performing hyperparameter scan...")
    if cluster:
        trials = MongoTrials(cluster, exp_key="exp1")
    else:
        trials = Trials()
    best = fmin(
        build_and_train_model,
        search_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
    )
    best_setup = space_eval(search_space, best)
    print("\n[+] Best scan setup:")
    pprint.pprint(best_setup)
    with open("%s/best-model.yaml" % folder, "w") as wfp:
        yaml.dump(best_setup, wfp, default_flow_style=False)
    log = "%s/hyperopt_log_{}.pickle".format(tm()) % folder
    with open(log, "wb") as wfp:
        print(f"[+] Saving trials in {log}")
        pickle.dump(trials.trials, wfp)
    return best_setup


def load_yaml(runcard_file):
    """Loads yaml runcard"""
    with open(runcard_file, "r") as stream:
        runcard = yaml.load(stream, Loader=yaml.FullLoader)
    for key, value in runcard.items():
        if ("hp." in str(value)) or ("None" == str(value)):
            runcard[key] = eval(value)
    return runcard


def build_and_train_model(setup):
    """Training model"""
    print("[+] Training model")
    if setup["model"] not in ("CNN", "CNNv2", "GCNN", "GCNNv2"):
        raise ValueError("Invalid input: choose one model at a time.")

    args = Args(**setup)

    loss, var_loss, _ = denoise.main(args)
    if setup["scan"]:
        res = {"loss": loss, "loss_variance": var_loss, "status": STATUS_OK}
    else:
        res = args
    return res


def main():
    """Parsing command line arguments"""
    parser = argparse.ArgumentParser(description="Train a generative model.")
    parser.add_argument(
        "--runcard", action="store", type=str, help="A yaml file with the setup."
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="The output folder"
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        dest="force",
        help="Overwrite existing files if present.",
    )
    parser.add_argument(
        "--hyperopt", default=None, type=int, help="Enable hyperopt scan."
    )
    parser.add_argument(
        "--cluster", default=None, type=str, help="Enable cluster scan."
    )
    args = parser.parse_args()

    # check input is coherent
    if not os.path.isfile(args.runcard):
        raise ValueError("Invalid runcard: not a file.")
    if args.force:
        print("WARNING: Running with --force option will overwrite existing model.")

    # prepare the output folder
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    elif args.force:
        shutil.rmtree(args.output)
        os.mkdir(args.output)
    else:
        raise Exception(f'{args.output} already exists, use "--force" to overwrite.')
    folder = args.output.strip("/")

    # load runcard
    setup = load_yaml(args.runcard)

    if args.hyperopt:
        setup["scan"] = True
        setup = run_hyperparameter_scan(setup, args.hyperopt, args.cluster, folder)
    setup["scan"] = False
    setup["out_name"] = "best_model"

    # build and train the model
    print("\n\n\n\n\nTrain best model")
    ARGS = build_and_train_model(setup)
    print("\nBest model analysis")
    loss, var_loss = analysis.main(ARGS)

    # write out a file with basic information on the run
    with open("%s/info.txt" % folder, "w") as f:
        print("# %s" % ARGS.model, file=f)
        print("# created on %s with the command:" % datetime.datetime.utcnow(), file=f)
        print("# " + " ".join(sys.argv), file=f)
        print("# final loss:\t%f +/- %f" % (loss, var_loss), file=f)

    # copy runcard to output folder
    shutil.copyfile(args.runcard, f"{folder}/input-runcard.json")

    # save the model to file
    # model.save(folder)


if __name__ == "__main__":
    START = tm()
    main()
    print("Program done in %f" % (tm() - START))
