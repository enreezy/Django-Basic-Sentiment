from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, HttpResponseRedirect																																				
from django.shortcuts import get_object_or_404, render, redirect
	
from django.http import Http404
from django.urls import reverse
from django.conf import settings
import nltk
from nltk.tokenize import word_tokenize
import random
import pickle


def index(request):
	return render(request,"home/index.html")

def sentiment(request):


	open_file = open("wordfeature5k.pickle","rb")
	word_features = pickle.load(open_file)
	open_file.close()


	def find_features(document):
	    words = word_tokenize(document)
	    features = {}
	    for w in word_features:
	        features[w] = (w in words)
	    return features

	open_file = open("naivebayesclassifier.pickle","rb")
	classifier = pickle.load(open_file)
	open_file.close()

	sentence = request.POST['sentence']

	result = classifier.classify(find_features(sentence))

	if result == "positive":
		return render(request, "home/index.html",{"sentence":sentence, "positive":"positive"})
	elif result == "negative":
		return render(request, "home/index.html",{"sentence":sentence, "negative":"negative"})




            

