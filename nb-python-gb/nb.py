#!/usr/bin/python

import math
import nb_classes

class NB:

	def __init__(self):
		self.spam_cat = nb_classes.nbCategory()
		self.ham_cat  = nb_classes.nbCategory()
	
	def data_append(self, tokens, cat):
		nbcat = None
		if cat=="spam":
			nbcat = self.spam_cat
		elif cat=="ham":
			nbcat = self.ham_cat
		
		for token in tokens:
			if token not in nbcat.words:
				nbcat.words[token]=1
			else:
				nbcat.words[token]+=1
		nbcat.weight+=1

	def data_process(self):
		#calcula as probabilidades das palavras em spam
		for (word,count) in self.spam_cat.words.items():
			#
			p  = float(count)+1/self.spam_cat.weight+2
			total = count
			if word in self.ham_cat.words:
				count+=self.ham_cat.words[word]
			p = p/( total+2/self.spam_cat.weight+self.ham_cat.weight+4 )
			#
			self.spam_cat.words_prob[word] = math.log(p)
		self.spam_cat.prob = math.log(float(self.spam_cat.weight)+2/self.spam_cat.weight+self.ham_cat.weight+4)
		self.spam_cat.none_log = math.log(2.0/self.spam_cat.weight+2)

		#
		#calcula as probabilidades das palavras em ham
		for (word,count) in self.ham_cat.words.items():
			#
			p  = float(count)+1/self.ham_cat.weight+2
			total = count
			if word in self.spam_cat.words:
				count+=self.spam_cat.words[word]
			p = p/( total+2/self.spam_cat.weight+self.ham_cat.weight+4 )
			#
			self.ham_cat.words_prob[word] = math.log(p)
		self.ham_cat.prob = math.log(float(self.ham_cat.weight)+2/self.spam_cat.weight+self.ham_cat.weight+4)
		self.ham_cat.none_log = math.log(2.0/self.ham_cat.weight+2)

	
	def prob(self, tokens, cat):
		nbcat = None
		if cat=="spam":
			nbcat = self.spam_cat
		elif cat=="ham":
			nbcat = self.ham_cat
		
		p = nbcat.prob
		count = 0
		for token in tokens:
			if token in nbcat.words_prob:
				p+=nbcat.words_prob[token]
			else:
				p+=nbcat.none_log
				count+=1
		return p
	
