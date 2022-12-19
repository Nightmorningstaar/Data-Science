import speech_recognition as sr 
import playsound
from gtts import gTTS  
import os  
import webbrowser
import wolframalpha   
num = 1

def process_text(input):
    try:
        if 'search' in input or 'play' in input:
            search_web(input)
            return

        elif "who are you" in input or "define yourself" in input or "What\'s your name" in input:
            speak = '''Hello, I am Claira. Your Personal Digital 
                    Assistant.I am here to make your life easier. You 
                    can command me to perform various tasks such as 
                    calculating sums or opening applications etcetra'''
            assistant_speaks(speak)
            return


	elif "who made you" in input or "created you" in input:
	    speak = '''I have been created by Samay, Sagar and Praveen 
                    for as a minor project for the semester.'''
            assistant_speaks(speak)
	    return

	elif "calculate" in input.lower():
            # write your wolframalpha app_id here
	    app_id = "rahulanon12345@gmail.com"
	    client = wolframalpha.Client(app_id)
	    indx = input.lower().split().index('calculate')
	    query = input.split()[indx + 1:]
	    res = client.query(' '.join(query))
	    answer = next(res.results).text
	    assistant_speaks("The answer is " + answer)
	    return

        elif "notes" in  input:
            assistant_speaks('Say the name of the note.')
            fname = get_audio() + str('.txt')
            assistant_speaks('Make the Note..')
            file_content = open(fname, 'a')
            text = get_audio().lower()
            file_content.write('' + text)
            file_content.close()
            assistant_speaks('The note has been saved')
        
        elif 'open' in input:
	    open_application(input.lower())
    	    return


	else:
	    assistant_speaks("I can search the web for you, Do you want                             to continue?")
	    ans = get_audio()
	    if 'yes' in str(ans) or 'yeah' in str(ans):
	        search_web(input)
	    else:
		return
    except :
        assistant_speaks("I don't understand, I can search the web for you, Do you want to continue?")
	ans = get_audio()
	if 'yes' in str(ans) or 'yeah' in str(ans):
	    search_web(input)

def search_web(input):
    if 'youtube' in input.lower(): 
        assistant_speaks("Opening in youtube") 
	indx = input.lower().split().index('youtube') 
	query = input.split()[indx + 1:] 
	os.system("firefox www.youtube.com/results?search_query=$query ")
	return

    elif 'wikipedia' in input.lower(): 
        assistant_speaks("Opening Wikipedia") 
	indx = input.lower().split().index('wikipedia') 
	query = input.split()[indx + 1:] 
	os.system("firefox en.wikipedia.org/wiki/$query")
        new += 1
	return

    else: 
        if 'google' in input:
            indx = input.lower().split().index('google') 
	    query = input.split()[indx + 1:]
	    os.system('chrome --no-sandbox $query')
	elif 'search' in input: 
            indx = input.lower().split().index('search') 
	    query = input.split()[indx + 1:] 
	    os.system('firefox $query &')
	else: 
            os.system("firefox &") 
        return

def open_application(input): 
    if "firefox" in input or "mozilla" in input: 
        assistant_speaks("Opening Mozilla Firefox") 
	os.system('firefox &') 
	return
    elif "chrome" in input:
        assistant_speaks("Opening Google Chrome")
        os.system('chrome --no-sandbox &')
        return
    elif "VLC" in input or "vlc" in input:
        assistant_speaks("Opening VLC Media Player")
        os.system('vlc &')
        return
    elif "word" in input or "notepad" in input: 
	assistant_speaks("Opening Text Editor") 
	os.system('gedit &') 
	return
    elif "excel" in input: 
	assistant_speaks("Opening Libre office Excel") 
	os.system('libreoffice &') 
	return
    else: 
	assistant_speaks("Application not available") 
	return

def assistant_speaks(output): 
    global num 
    num += 1
    print("Claira : ", output) 
    toSpeak = gTTS(text = output, lang ='en', slow = False) 
    file = str(num)+".mp3" 
    toSpeak.save(file) 	
    playsound.playsound(file, True) 
    os.remove(file) 

def get_audio(): 
    rObject = sr.Recognizer() 
    audio = '' 
    with sr.Microphone() as source: 
        print("\n\t\t\tGive The Command ....") 
	audio = rObject.listen(source, phrase_time_limit = 5) 
        print("\t\t\tStop .....")  
        try: 
            text = rObject.recognize_google(audio, language ='en-US') 
	    print("You : ", text) 
	    return text 
        except: 
	    assistant_speaks("Could not understand your audio, PLease try again !") 
	    return 0


# Driver Code 
if __name__ == "__main__": 
    owner = ["Samay", "Sagar", "Praveen"]
    assistant_speaks("What's your name, Human?") 
    name ='Human'
    name = get_audio() 
    if name in owner:
        assistant_speaks("Hello Boss, It\'s been a long time.")
    else:
        assistant_speaks("Hello, " + name + '.') 
    while(1): 
        assistant_speaks("What can i do for you?") 

	text = get_audio()
	if text == 0: 
	    continue
        if "exit" in str(text) or "bye" in str(text) or "shutdown" in str(text) or "you can rest now" in str(text) or "sleep" in str(text): 
            assistant_speaks("Ok bye, "+ name+'.')
	    break
        process_text(text)



# -> gTTS – Google Text To Speech, for converting the given text to speech
# -> speech_recognition – for recognizing the voice command and converting to text
# -> selenium – for web based work from browser
# -> wolframalpha – for calculation given by user
# -> playsound – for playing the saved audio file.
# -> pyaudio – for voice engine in python
# import os # to save/open files