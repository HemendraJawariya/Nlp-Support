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

        # Load only the selected pipeline to reduce memory usage
        if task == 'sentiment':
            sentiment_pipeline = pipeline('sentiment-analysis')
            sent = sentiment_pipeline(user_input)

        elif task == 'generation':
            generation_pipeline = pipeline('text-generation')
            gen = generation_pipeline(user_input)

        elif task == 'translation':
            translation_pipeline = pipeline('translation', model="Helsinki-NLP/opus-mt-fr-en")
            trans = translation_pipeline(user_input)

        elif task == 'summarization':
            summarization_pipeline = pipeline('summarization')
            summ = summarization_pipeline(user_input)

        elif task == 'named_entity_recognition':
            NER_pipeline = pipeline("ner", grouped_entities=True)
            ner = NER_pipeline(user_input)

    return render_template('1_pipeline.html', sent=sent, gen=gen, trans=trans, ner=ner, summ=summ)

# Bind to the port Render provides
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
