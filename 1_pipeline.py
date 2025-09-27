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
            sentiment_pipeline = pipeline('sentiment-analysis', model='distilbert/distilbert-base-uncased-finetuned-sst-2-english')
            sent = sentiment_pipeline(user_input)

        elif task == 'generation':
            generation_pipeline = pipeline('text-generation', model='gpt2')
            gen = generation_pipeline(user_input)

        elif task == 'translation':
            translation_pipeline = pipeline('translation', model='Helsinki-NLP/opus-mt-fr-en')
            trans = translation_pipeline(user_input)

        elif task == 'summarization':
            summarization_pipeline = pipeline('summarization', model='facebook/bart-large-cnn')
            summ = summarization_pipeline(user_input)

        elif task == 'named_entity_recognition':
            NER_pipeline = pipeline('ner', model='dslim/bert-base-NER', grouped_entities=True)
            ner = NER_pipeline(user_input)

    return render_template('1_pipeline.html', sent=sent, gen=gen, trans=trans, ner=ner, summ=summ)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 4000))
    app.run(debug=False, host='0.0.0.0', port=port)

