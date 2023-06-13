import pickle
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from torch import nn
import nltk
import re
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F



def load_data(path):
    df = pd.read_csv(path, sep='\t')
    return df

# defining global variables
stops = set(stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()


# take into account the lemmatization and stemming
def review_to_words(review):
	# remove HTML tags
	clean_review = re.sub(r'<.*?>', '', review)
	# remove non-letters
	letters = re.sub('[^a-zA-Z]', ' ', clean_review)
	words = letters.lower().split()
	without_stops = [w for w in words if not w in stops]
	lemmatization_words = [lemmatizer.lemmatize(w) for w in without_stops]
	return ' '.join(lemmatization_words)

class NeuralNet(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes):
		super(NeuralNet, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.hidden = nn.Linear(hidden_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, num_classes)

	def forward(self, x):
		out = self.fc1(x)
		out = F.relu(out)
		out = self.hidden(out)
		out = F.relu(out)
		out = self.fc2(out)
		return out


def plot_confusion_matrix(data, classes):
	plt.figure(figsize=(10, 10))
	plt.imshow(data, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title('Confusion matrix')
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes)
	plt.yticks(tick_marks, classes)
	thresh = data.max() / 2.
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			value = data[i, j]
			if value > thresh:
				color = "white"
			else:
				color = "black"
			plt.text(j, i, value, horizontalalignment="center", color=color)
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')


def predict_ratings_by_reviews(num_classes = 11):
	"""
	Insert 3 classes.
	Default is 10 classes.
	    """
	full_test_set = load_data('drugsComTest_raw.tsv')
	full_train_set = load_data('drugsComTrain_raw.tsv')

	# check the head of the data
	print(full_train_set.head())
	print(full_test_set.head())

	# check the shape of the data
	print(f"train set shape: {full_train_set.shape}")
	print(f"test set shape: {full_test_set.shape}")

	# check the columns of the data
	print(f"train set columns: {full_train_set.columns}")
	print(f"test set columns: {full_test_set.columns}")

	# check if there are any null values
	print(f"train set null values: \n{full_train_set.isnull().sum()}")
	print(f"test set null values: \n{full_test_set.isnull().sum()}")


	# check if there are any duplicates
	print(f"train set duplicates: {full_train_set.duplicated().sum()}")
	print(f"test set duplicates: {full_test_set.duplicated().sum()}")

	# check how 5 random reviews look like
	print(full_train_set['review'].sample(5).values)



	# apply the function "review_to_words" to the train set and test set
	full_train_set['review'] = full_train_set['review'].apply(review_to_words)
	full_test_set['review'] = full_test_set['review'].apply(review_to_words)

	if num_classes == 3:
		# apply 3 classes to the ratings (0-4 -> 0, 5-6 -> 1, 7-10 -> 2)
		full_train_set['rating'] = full_train_set['rating'].apply(lambda x: 0 if x <= 4 else 1 if x < 7 else 2)
		full_test_set['rating'] = full_test_set['rating'].apply(lambda x: 0 if x <= 4 else 1 if x < 7 else 2)
	X_train = full_train_set['review']
	y_train = full_train_set['rating']
	X_test = full_test_set['review']
	y_test = full_test_set['rating']


	# check the distribution of the ratings in the train set
	plt.hist(y_train, bins=10)
	plt.xticks(np.arange(1, 10, 1))
	plt.xlabel('rating')
	plt.ylabel('count')
	plt.title('distribution of the ratings in the train set')
	plt.show()


	# print the maximum length of a review and the mean length of a review
	print(f"max length of a review in the train test: {max(X_train.apply(lambda x: len(x)))}")
	print(f"mean length of a review in the train test: {X_train.apply(lambda x: len(x)).mean()}")
	# print the 80% percentile of the length of a review
	print(f"80% percentile of the length of a review in the train test: {X_train.apply(lambda x: len(x)).quantile(0.8)}")
	######### ====>>>> 6433, 265, 410


	# calculate the number of unique words in the train set
	unique_words = set()
	for review in X_train:
		unique_words.update(review.split())
	print(f"number of unique words in the train set: {len(unique_words)}")
	######### ====>>>> 42000 unique words

	# split the train set to train set and validation set
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=42)

	# CountVectorizer - convert the reviews to vectors of numbers (bag of words)
	max_features = 5000
	vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1, 2))
	X_train = vectorizer.fit_transform(X_train).toarray()
	X_test = vectorizer.transform(X_test).toarray()
	X_val = vectorizer.transform(X_val).toarray()



	print(f"shape of the train set: {X_train.shape}")
	print(f"shape of the test set: {X_test.shape}")
	print(f"shape of the validation set: {X_val.shape}")


	X_train = torch.from_numpy(X_train).float()
	y_train = torch.from_numpy(y_train.values)
	X_test = torch.from_numpy(X_test).float()
	y_test = torch.from_numpy(y_test.values)
	X_val = torch.from_numpy(X_val).float()
	y_val = torch.from_numpy(y_val.values)


	# create the neural network

	# define the hyperparameters
	# num_classes defined in the function parameters
	input_size = max_features
	hidden_size = 250
	num_epochs = 3
	if num_classes == 11:
		num_epochs = 8
	batch_size = 150
	learning_rate = 1e-3


	model = NeuralNet(input_size, hidden_size, num_classes)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	# train the model
	model.train()
	current_loss = 0
	avg_loss = []
	acc_val = []
	acc_train = []
	for epoch in range(num_epochs):
		model.train()
		for i in range(0, len(X_train), batch_size):
			reviews = X_train[i:i+batch_size]
			labels = y_train[i:i+batch_size]
			outputs = model(reviews)
			if num_classes == 11:
				labels = labels.long()
			loss = criterion(outputs, labels)
			accuracy = accuracy_score(labels.detach().numpy(), outputs.detach().numpy().argmax(axis=1))
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			current_loss += loss.item()

			if i % 2000 == 0:   # print statistics
				print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(X_train)}], Loss: {loss.item():.4f}, Accuracy: {(accuracy*100):.4f} %')
		acc_train.append(accuracy)
		avg_loss.append(current_loss / len(X_train))
		current_loss = 0
		model.eval()
		# calculate the accuracy on the validation set
		with torch.no_grad():
			correct = 0
			total = 0
			for i in range(0, len(X_val), batch_size):
				reviews = X_val[i:i + batch_size]
				labels = y_val[i:i + batch_size]
				outputs = model(reviews)
				_, predicted = torch.max(outputs.data, 1)
				total += len(labels)
				correct += (predicted == labels).sum().item()
			print(f'Accuracy of the network on the {len(X_val)} validation reviews: {(100 * correct / total):.4f} % in epoch {epoch+1}')
			acc_val.append(correct / total)

	print('Finished Training')

	# plot the loss
	plt.plot([i for i in range(1, num_epochs + 1)], avg_loss, label='loss')
	plt.xlabel('epoch')
	plt.ylabel('train_loss')
	plt.title('train_loss Vs epochs')
	plt.show()

	# plot the accuracy
	plt.plot([i for i in range(1, num_epochs + 1)], acc_train, label='train_accuracy')
	plt.plot([i for i in range(1, num_epochs + 1)], acc_val, label='validation_accuracy')
	plt.ylim(0, 1)
	plt.yticks(np.arange(0, 1, 0.05))
	plt.xlabel('epoch')
	plt.ylabel('accuracy')
	plt.title('accuracy Vs epochs')
	plt.legend()
	plt.show()



	# test the model
	model.eval()
	all_labels = []
	all_predicted = []
	with torch.no_grad():
		correct = 0
		total = 0
		for i in range(0, len(X_test), batch_size):
			reviews = X_test[i:i+batch_size]
			labels = y_test[i:i+batch_size]
			outputs = model(reviews)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
			all_labels.extend(labels)
			all_predicted.extend(predicted)
		cnf_matrix = confusion_matrix(all_labels, all_predicted)
		if num_classes == 3:
			plot_confusion_matrix(cnf_matrix, classes=['0', '1', '2'])
		else:
			plot_confusion_matrix(cnf_matrix, classes=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])

		print(f'Accuracy of the network on the test set: {100 * correct / total} %')
		plt.show()

	# save the model
	if num_classes == 3:
		torch.save(model, 'final_model3')
	else:
		torch.save(model, 'final_model10')

	# save the vectorizer
	with open('vectorizer.pickle', 'wb') as handle:
		pickle.dump(vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)




def example(reviews, model_name):
	vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))
	new_reviews = vectorizer.transform(reviews)
	new_reviews = torch.from_numpy(new_reviews.toarray()).float()
	model = torch.load(model_name)
	with torch.no_grad():
		outputs = model(new_reviews)
		_, predicted = torch.max(outputs.data, 1)
		print('Predicted ratings: {}'.format(predicted))




def main():
	predict_ratings_by_reviews()
    # predict_ratings_by_reviews()
	# reviews = [""]
	# example(reviews, 'final_model10')

if __name__ == '__main__':
    main()

