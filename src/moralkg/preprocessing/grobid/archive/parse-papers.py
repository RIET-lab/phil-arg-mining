
# Extract the references from all pdfs downloaded.

# https://github.com/kermitt2/grobid_client_python
# https://grobid.readthedocs.io/en/latest/Grobid-service/

from grobid_client.grobid_client import GrobidClient

client = GrobidClient(config_path="/opt/extra/avijit/projects/moralkg/data/scripts/grobid/config.json")
client.process("processReferences", 
               "/opt/extra/avijit/projects/moralkg/data/pdfs",
               output="/opt/extra/avijit/projects/moralkg/data/grobid",
               n=64,
               force=False,
               verbose=True)
