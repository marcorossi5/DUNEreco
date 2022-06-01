: '
This scripts downloads and extracts the following Zenodo dataset.
Zenodo dataset DOI: https://doi.org/10.5281/zenodo.6599305

Running the script creates a tarball at
`examples/dunetpc_inspired_v08_p2GeV_rawdigits.tar.gz` and extracts the model
checkpoints folder at `dunedn_checkpoints.tar.gz`.

Usage: `bash download_dataset.sh`
'
# create temporary directory
mkdir tmp

# sample event
wget https://zenodo.org/record/6599305/files/dunetpc_inspired_v08_p2GeV_rawdigits.tar.gz -P tmp
mv tmp/dunetpc_inspired_v08_p2GeV_rawdigits.tar.gz examples/

# models checkpoints
wget https://zenodo.org/record/6599305/files/dunedn_checkpoints.tar.gz -P tmp
tar -xzvf tmp/dunedn_checkpoints.tar.gz -C .

# delete temprary directory
rm -rf tmp