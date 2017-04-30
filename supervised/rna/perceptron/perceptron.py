# Implementação Perceptron
import random

class Perceptron:

	def __init__(self, amostras, saidas, taxa_aprendizado=0.1, epocas=1000, limiar=-1):
		self.amostras = amostras
		self.saidas = saidas
		self.taxa_aprendizado = taxa_aprendizado
		self.epocas = epocas
		self.limiar = limiar
		self.n_amostras = len(amostras)
		self.n_atributos = len(amostras[0])
		self.pesos = []

	def treinar(self):

		for amostra in self.amostras:
			amostra.insert(0, -1)
		
		for i in range(self.n_atributos):
			self.pesos.append(random.random())

		self.pesos.insert(0, self.limiar)
		n_epocas = 0 # contador de épocas

		while True:
			erro = False # erro inicialmente inexiste

			for i in range(self.n_amostras):
				u = 0 # Somatório do produto dos pesos dos sinais de entrada pelos pesos
				for j in range(self.n_atributos + 1):
					u += self.pesos[j] * self.amostras[i][j]
				y = self.degrau(u) # obtém a saída da rede

				# verifica se a saida da rede é diferente da saída desejada
				if y != self.saidas[i]:
					# calcula o erro
					erro_aux = self.saidas[i] - y
					# faz o ajuste dos pesps para cada elemento da amostra
					for j in range(self.n_atributos + 1):
						self.pesos[j] = self.pesos[j] + self.taxa_aprendizado * erro_aux * self.amostras[i][j]
					erro = True # erro ainda existe
			n_epocas += 1 # incrementa o número de épocas
			# critério de parada
			if not erro or n_epocas > self.epocas:
				break
		print(self.pesos)
		print(n_epocas)

	def degrau(self, u):
		if u >= 0:
			return 1
		return 0

# OR
entradas = [[0, 0], [0, 1], [1, 0], [1, 1]]
saidas = [0, 1, 1, 1]
rede = Perceptron(entradas, saidas)
rede.treinar()