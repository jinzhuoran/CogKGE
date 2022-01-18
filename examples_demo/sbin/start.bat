cd ..
pip install hupper waitress
hupper -m waitress --listen=*:5050 --threads 5 flask_cogkge_demo:app
