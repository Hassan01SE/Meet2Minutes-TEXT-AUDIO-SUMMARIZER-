import warnings
warnings.filterwarnings("ignore")
from flask import Flask, render_template, Response,request
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from summarizer import Summarizer
# import speech_recognition as sr 
import moviepy.editor as mp
from time import sleep
import json
import pprint
import requests
from googletrans import Translator


app=Flask(__name__)

def load_models():
    global model
    global tokenizer
    global bert_model
    model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')
    print('model downloaded')
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    print('model downloaded')
    bert_model = Summarizer()
    print('model downloaded')

API_key = "057a0adabc0e4d25ba5e9db2d155c8b0" 

headers = headers = {
    'authorization': API_key, 
    'content-type': 'application/json',
}

upload_endpoint = 'https://api.assemblyai.com/v2/upload'
transcription_endpoint = "https://api.assemblyai.com/v2/transcript"    

# def upload_to_AssemblyAI(audio_file):

#     transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
#     upload_endpoint = 'https://api.assemblyai.com/v2/upload'

#     print('uploading')
#     upload_response = requests.post(
#         upload_endpoint,
#         headers=headers, data=audio_file
#     )

#     audio_url = upload_response.json()['upload_url']
#     print('done')

#     json = {
#         "audio_url": audio_url,
#         "iab_categories": True,
#         "auto_chapters": True
#     }

#     response = requests.post(transcript_endpoint, json=json, headers=headers)
#     print(response.json())

#     polling_endpoint = transcript_endpoint + "/" + response.json()['id']
#     return polling_endpoint



def upload(file_path):

    def read_audio(file_path):

        with open(file_path, 'rb') as f:
            while True:
                data = f.read(5_242_880)
                if not data:
                    break
                yield data

    upload_response =  requests.post(upload_endpoint, 
                                     headers=headers, 
                                     data=read_audio(file_path))

    return upload_response.json().get('upload_url')

def transcribe(upload_url): 

    json = {"audio_url": upload_url, "auto_chapters": True,"iab_categories": True }
    
    response = requests.post(transcription_endpoint, json=json, headers=headers)
    transcription_id = response.json()['id']

    return transcription_id

def get_result(transcription_id): 

    current_status = "queued"

    endpoint = f"https://api.assemblyai.com/v2/transcript/{transcription_id}"

    while current_status not in ("completed", "error"):
        
        response = requests.get(endpoint, headers=headers)
        current_status = response.json()['status']
        
        if current_status in ("completed", "error"):
            return response.json()
        else:
            sleep(10)
          
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/audio", methods=['post'])
def audio():
    try:
        upload_url = upload(request.form["file"])
    except:
        clip = mp.VideoFileClip(request.form["file"])  
        clip.audio.write_audiofile(r"converted")    
        upload_url = upload('converted.mp3')
    transcription_id = transcribe(upload_url)
    response = get_result(transcription_id)
    summary = response["text"][0:500]
    headline = response["chapters"][0]['headline']
    try:
        language = str(request.form["lang1"])
        translator = Translator()
        summarynew = translator.translate(summary,dest = language)
        head_new = translator.translate(headline,dest = language)
        headline = head_new.text
        summary = summarynew.text
    except:
        pass    
    themes = []
    # polling_endpoint = upload_to_AssemblyAI(upload_url)
    # polling_response = requests.get(polling_endpoint, headers=headers)
    # status = polling_response.json()['status']
    categories = response['iab_categories_result']['summary']
    # if status == 'completed':
    for cat in categories:
            themes.append(cat)
    print(categories)

    return render_template("result2.html", summary = summary, headline = headline , themes = themes)


@app.route("/model", methods=['post'])
def model():
    try:
        text = str(request.form['text'])
    except:
        pass
        # clip = mp.VideoFileClip(request.form["file"])  
        # clip.audio.write_audiofile(r"converted.wav")
        # r = sr.Recognizer()
        # audio = sr.AudioFile("converted.wav")
        # with audio as source:
        #     audio_file = r.record(source)
        # result = r.recognize_google(audio_file)
        # text = result
    
    tokens_input = tokenizer.encode("summarize: "+text, return_tensors='pt', 
                                max_length=tokenizer.model_max_length, 
                                truncation=True)
    summary_ids = model.generate(tokens_input, min_length=80,
                             max_length=150,
                             length_penalty=20, 
                             num_beams=2)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    t5_tokens_len = tokenizer.model_max_length
    
    ext_summary = bert_model(text, ratio=0.5)
    bert_token_len = len(tokenizer(ext_summary)['input_ids'])
    print('t5: ',t5_tokens_len,'\n','bert: ',bert_token_len)   
    tokens_input_2 = tokenizer.encode("summarize: "+ext_summary, return_tensors='pt', 
                                max_length=tokenizer.model_max_length, 
                                truncation=True)
    summary_ids_2 = model.generate(tokens_input_2, min_length=80,
                             max_length=150,
                             length_penalty=20, 
                             num_beams=2)

    summary_2 = tokenizer.decode(summary_ids_2[0], skip_special_tokens=True)

    try:
        language = str(request.form["lang"])
        translator = Translator()
        summarynew = translator.translate(summary,dest = language)
        summarynew1 = translator.translate(ext_summary,dest = language)
        summarynew2 = translator.translate(summary_2,dest = language)
        summary = summarynew.text
        ext_summary = summarynew1.text
        summary_2 = summarynew2.text
    except:
        pass    
    return render_template("result.html",summary = summary, ext_summary = ext_summary, summary_2 = summary_2)
    
  
    

if __name__ == '__main__':
    load_models()
    app.run(debug=True)