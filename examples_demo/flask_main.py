# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request
from flask_cors import *
import json

def predict(sentense):
    return {
      "sentence": "Davies is leaving to become chairman of the London School of Economics , one of the best-known parts of the University of London .",
      "golden-event-mentions": [
         {
            "trigger": {
               "text": "leaving",
               "start": 2,
               "end": 3
            },
            "arguments": [
               {
                  "role": "Person",
                  "entity-type": "PER:Individual",
                  "text": "Davies",
                  "start": 0,
                  "end": 1
               }
            ],
            "event_type": "Personnel:End-Position"
         },
         {
            "trigger": {
               "text": "become",
               "start": 4,
               "end": 5
            },
            "arguments": [
               {
                  "role": "Person",
                  "entity-type": "PER:Individual",
                  "text": "Davies",
                  "start": 0,
                  "end": 1
               },
               {
                  "role": "Entity",
                  "entity-type": "ORG:Educational",
                  "text": "the London School of Economics",
                  "start": 7,
                  "end": 12
               },
               {
                  "role": "Position",
                  "entity-type": "Job-Title",
                  "text": "chairman of the London School of Economics, one of the best-known parts of the University of London",
                  "start": 5,
                  "end": 23
               }
            ],
            "event_type": "Personnel:Start-Position"
         }
      ],
      "golden-entity-mentions": [
         {
            "text": "Davies",
            "entity-type": "PER:Individual",
            "head": {
               "text": "Davies",
               "start": 0,
               "end": 1
            },
            "entity_id": "AFP_ENG_20030401.0476-E1-2",
            "start": 0,
            "end": 1
         },
         {
            "text": "chairman of the London School of Economics, one of the best-known parts of the University of London",
            "entity-type": "PER:Individual",
            "head": {
               "text": "chairman",
               "start": 5,
               "end": 6
            },
            "entity_id": "AFP_ENG_20030401.0476-E1-31",
            "start": 5,
            "end": 23
         },
         {
            "text": "the London School of Economics",
            "entity-type": "ORG:Educational",
            "head": {
               "text": "London School of Economics",
               "start": 8,
               "end": 12
            },
            "entity_id": "AFP_ENG_20030401.0476-E2-4",
            "start": 7,
            "end": 12
         },
         {
            "text": "one of the best-known parts of the University of London",
            "entity-type": "ORG:Educational",
            "head": {
               "text": "one",
               "start": 13,
               "end": 14
            },
            "entity_id": "AFP_ENG_20030401.0476-E2-5",
            "start": 13,
            "end": 23
         },
         {
            "text": "the University of London",
            "entity-type": "ORG:Educational",
            "head": {
               "text": "University of London",
               "start": 20,
               "end": 23
            },
            "entity_id": "AFP_ENG_20030401.0476-E4-6",
            "start": 19,
            "end": 23
         },
         {
            "text": "chairman of the London School of Economics, one of the best-known parts of the University of London",
            "entity-type": "Job-Title",
            "head": {
               "text": "chairman of the London School of Economics, one of the best-known parts of the University of London",
               "start": 5,
               "end": 23
            },
            "entity_id": "AFP_ENG_20030401.0476-V3-1",
            "start": 5,
            "end": 23
         }
      ]
   }


app = Flask(__name__, static_url_path='')
app.config['JSON_AS_ASCII'] = False
CORS(app,resources={r"/*": {"origins":"*"}}, send_wildcard=True, supports_credentials=True)
# 只接受POST方法访问


@app.route('/', methods=["GET"])
def index():
    return app.send_static_file('test.html')


@app.route('/event_extraction.json', methods=["GET", "POST"])
def event_extraction():
    sentence = "As well as previously holding senior positions at Barclays Bank , BZW and Kleinwort Benson , McCarthy was formerly a top civil servant at the Department of Trade and Industry ."
    if request.method == "POST":
        sentence = request.form['sentence']
    else:
        sentence = request.args['sentence']

    return jsonify(predict(sentence))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
