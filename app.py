from flask import Flask, render_template, request
from transformers import pipeline
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    sent = gen = trans = ner = summ = None

    if request.method == 'POST':
        user_input = request.form.get('user_input')
        task = request.form.get('task')

        if task == 'sentiment':
            sent = pipeline('sentiment-analysis', model='distilbert/distilbert-base-uncased-finetuned-sst-2-english')(user_input)

        elif task == 'generation':
            gen = pipeline('text-generation', model='distilgpt2')(user_input)

        elif task == 'translation':
            trans = pipeline('text2text-generation', model='t5-small')(f"translate French to English: {user_input}")

        elif task == 'summarization':
            summ = pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')(user_input)

        elif task == 'named_entity_recognition':
            ner = pipeline('ner', model='dslim/bert-base-NER', grouped_entities=True)(user_input)

    return render_template('1_pipeline.html', sent=sent, gen=gen, trans=trans, ner=ner, summ=summ)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
