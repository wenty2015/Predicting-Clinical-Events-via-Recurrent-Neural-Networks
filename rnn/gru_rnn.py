#################################################################
# Code is folked from doctor ai, written by Edward Choi (mp2893@gatech.edu)
# modified by Wen Qin, debugging and adding no-tune embedding, topic feature
#################################################################

import sys, random
import numpy as np
import cPickle as pickle
from collections import OrderedDict
import argparse

import theano
import theano.tensor as T
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

def unzip(zipped):
	new_params = OrderedDict()
	for key, value in zipped.iteritems():
		new_params[key] = value.get_value()
	return new_params

def numpy_floatX(data):
	return np.asarray(data, dtype=config.floatX)

def load_embedding(infile):
	Wemb = np.array(pickle.load(open(infile, 'rb'))).astype(config.floatX)
	return Wemb

# initialization of network
# return parameters of each layer
def init_params(options):
	params = OrderedDict()
	timeFile = options['timeFile']
	topicFile = options['topicFile']
	topicSize = options['topicSize']

	# PCA for embedding
	embFile = options['embFile']
	embSize = options['embSize']

	inputDimSize = options['inputDimSize']
	numClass = options['numClass']

	if len(embFile) > 0:
		print 'using external code embedding (PCA)'
		params['W_emb'] = load_embedding(embFile)
		embSize = params['W_emb'].shape[1]
	else:
		if options['embFineTune']:
			print 'using randomly initialized code embedding'
			params['W_emb'] = np.random.uniform(-0.01, 0.01, (inputDimSize, embSize)).astype(config.floatX)
			params['b_emb'] = np.zeros(embSize).astype(config.floatX)

		else:
			print 'use original input'
			params['W_emb'] = np.eye(inputDimSize).astype(config.floatX)
			embSize = inputDimSize

	prevDimSize = embSize
	if len(timeFile) > 0: prevDimSize += 1
	#We need to consider an extra dimension for the duration information
	if len(topicFile) > 0: prevDimSize += topicSize

	# set up the structure of network
	for count, hiddenDimSize in enumerate(options['hiddenDimSize']):
		params['W_'+str(count)] = np.random.uniform(-0.01, 0.01, (prevDimSize, hiddenDimSize)).astype(config.floatX)
		params['W_r_'+str(count)] = np.random.uniform(-0.01, 0.01, (prevDimSize, hiddenDimSize)).astype(config.floatX)
		params['W_z_'+str(count)] = np.random.uniform(-0.01, 0.01, (prevDimSize, hiddenDimSize)).astype(config.floatX)
		params['U_'+str(count)] = np.random.uniform(-0.01, 0.01, (hiddenDimSize, hiddenDimSize)).astype(config.floatX)
		params['U_r_'+str(count)] = np.random.uniform(-0.01, 0.01, (hiddenDimSize, hiddenDimSize)).astype(config.floatX)
		params['U_z_'+str(count)] = np.random.uniform(-0.01, 0.01, (hiddenDimSize, hiddenDimSize)).astype(config.floatX)
		params['b_'+str(count)] = np.zeros(hiddenDimSize).astype(config.floatX)
		params['b_r_'+str(count)] = np.zeros(hiddenDimSize).astype(config.floatX)
		params['b_z_'+str(count)] = np.zeros(hiddenDimSize).astype(config.floatX)
		prevDimSize = hiddenDimSize

	# set up output layer
	params['W_output'] = np.random.uniform(-0.01, 0.01, (prevDimSize, numClass)).astype(config.floatX)
	params['b_output'] = np.zeros(numClass).astype(config.floatX)

	return params

# share parameters
def init_tparams(params, options):
	tparams = OrderedDict()
	for key, value in params.iteritems():
		if not options['embFineTune'] and key == 'W_emb': continue
		tparams[key] = theano.shared(value, name=key)
	return tparams

def dropout_layer(state_before, use_noise, trng, dropout_rate):
	proj = T.switch(use_noise, (state_before * trng.binomial(state_before.shape, p=dropout_rate, n=1, dtype=state_before.dtype)), state_before * 0.5)
	return proj

def gru_layer(tparams, emb, layerIndex, hiddenDimSize, mask=None):
	timesteps = emb.shape[0]
	if emb.ndim == 3: n_samples = emb.shape[1]
	else: n_samples = 1

	W_rx = T.dot(emb, tparams['W_r_'+layerIndex])
	W_zx = T.dot(emb, tparams['W_z_'+layerIndex])
	Wx = T.dot(emb, tparams['W_'+layerIndex])

	def stepFn(stepMask, wrx, wzx, wx, h):
		r = T.nnet.sigmoid(wrx + T.dot(h, tparams['U_r_'+layerIndex]) + tparams['b_r_'+layerIndex])
		z = T.nnet.sigmoid(wzx + T.dot(h, tparams['U_z_'+layerIndex]) + tparams['b_z_'+layerIndex])
		h_tilde = T.tanh(wx + T.dot(r*h, tparams['U_'+layerIndex]) + tparams['b_'+layerIndex])
		h_new = z * h + ((1. - z) * h_tilde)
		h_new = stepMask[:, None] * h_new + (1. - stepMask)[:, None] * h
		return h_new

	results, updates = theano.scan(fn=stepFn, sequences=[mask,W_rx,W_zx,Wx], outputs_info=T.alloc(numpy_floatX(0.0), n_samples, hiddenDimSize), name='gru_layer'+layerIndex, n_steps=timesteps)

	return results

def build_model(tparams, options, W_emb=None):
	trng = RandomStreams(123)
	use_noise = theano.shared(numpy_floatX(0.))
	if len(options['timeFile']) > 0: useTime = True
	else: useTime = False

	if len(options['topicFile']) > 0: useTopics = True
	else: useTopics = False

	# 3-d ndarray, [time,sample,code]
	x = T.tensor3('x', dtype=config.floatX)
	y = T.tensor3('y', dtype=config.floatX)
	# 3-d ndarray, [time,sample]
	t = T.matrix('t', dtype=config.floatX)
	if useTopics:
		topics = T.tensor3('topics', dtype=config.floatX)
	else:
		topics = None

	t_label = T.matrix('t_label', dtype=config.floatX)
	mask = T.matrix('mask', dtype=config.floatX)
	lengths = T.vector('lengths', dtype=config.floatX)

	n_timesteps = x.shape[0]
	n_samples = x.shape[1]

	if options['embFineTune']:
		# tune W_emb during training
		emb = T.tanh(T.dot(x, tparams['W_emb']) + tparams['b_emb'])
	else:
		# todo- apply pca embedding
		emb = T.dot(x, W_emb)

	# add the time element to the input
	if useTime:
		emb = T.concatenate([t.reshape([n_timesteps,n_samples,1]), emb], axis=2)

	# add topics to the input
	if useTopics:
		emb = T.concatenate([topics, emb], axis=2)

	inputVector = emb
	for i, hiddenDimSize in enumerate(options['hiddenDimSize']):
		memories = gru_layer(tparams, inputVector, str(i), hiddenDimSize, mask=mask)
		memories = dropout_layer(memories, use_noise, trng, options['dropout_rate'])
		inputVector = memories

	def softmaxStep(memory2d):
		return T.nnet.softmax(T.dot(memory2d, tparams['W_output']) + tparams['b_output'])

	logEps = options['logEps']
	results, updates = theano.scan(fn=softmaxStep, sequences=[inputVector], \
									outputs_info=None, name='softmax_layer', n_steps=n_timesteps)

	# use mask to get results for input sequence only
	results = results * mask[:,:,None]
	cross_entropy = -(y * T.log(results + logEps) + (1. - y) * T.log(1. - results + logEps))
	# sum of cross_entropy for each sample
	prediction_loss = cross_entropy.sum(axis=2).sum(axis=0) / lengths

	# regularization
	cost = T.mean(prediction_loss) + options['L2_output'] * (tparams['W_output'] ** 2).sum()

	if useTime: return use_noise, x, y, t, topics, mask, lengths, cost
	else: return use_noise, x, y, topics, mask, lengths, cost

def adadelta(tparams, grads, x, y, mask, lengths, cost, options, t=None, topics=None, t_label=None):
	zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_grad' % k) for k, p in tparams.iteritems()]
	running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rup2' % k) for k, p in tparams.iteritems()]
	running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rgrad2' % k) for k, p in tparams.iteritems()]

	zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
	rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

	if len(options['timeFile']) > 0 and len(options['topicFile'])>0:
		f_grad_shared = theano.function([x, y, t, topics, mask, lengths], cost, updates=zgup + rg2up, name='adadelta_f_grad_shared')
	elif len(options['timeFile']) > 0 and len(options['topicFile']) == 0:
		f_grad_shared = theano.function([x, y, t, mask, lengths], cost, updates=zgup + rg2up, name='adadelta_f_grad_shared')
	elif len(options['timeFile']) == 0 and len(options['topicFile'])>0:
		f_grad_shared = theano.function([x, y, topics, mask, lengths], cost, updates=zgup + rg2up, name='adadelta_f_grad_shared')
	else:
		f_grad_shared = theano.function([x, y, mask, lengths], cost, updates=zgup + rg2up, name='adadelta_f_grad_shared')

	updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
	ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
	param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

	f_update = theano.function([], [], updates=ru2up + param_up, on_unused_input='ignore', name='adadelta_f_update')

	return f_grad_shared, f_update

def padMatrixWithTime(seqs, labels, times, options, topics=None):
	lengths = np.array([len(seq) for seq in seqs]) - 1
	n_samples = len(seqs)
	maxlen = np.max(lengths)
	inputDimSize = options['inputDimSize']
	numClass = options['numClass']
	useTopics = options['useTopics']

	x = np.zeros((maxlen, n_samples, inputDimSize)).astype(config.floatX)
	y = np.zeros((maxlen, n_samples, numClass)).astype(config.floatX)
	t = np.zeros((maxlen, n_samples)).astype(config.floatX)
	mask = np.zeros((maxlen, n_samples)).astype(config.floatX)

	if useTopics:
		topicSize = options['topicSize']
		top = np.zeros((maxlen, n_samples, topicSize)).astype(config.floatX)
		for idx, topic in enumerate(topics):
			for tvec, tvalue in zip(top[:lengths[idx], idx,:], topic[:-1]):
				tvec = tvalue.copy()

	for idx, (seq,time,label) in enumerate(zip(seqs,times,labels)):
		for xvec, subseq in zip(x[:,idx,:], seq[:-1]):
			xvec[subseq] = 1.
		for yvec, subseq in zip(y[:,idx,:], label[1:]):
			yvec[subseq] = 1.
		mask[:lengths[idx], idx] = 1.
		t[:lengths[idx], idx] = time[:-1]

	lengths = np.array(lengths, dtype=config.floatX)
	if options['useLogTime']:
		t = np.log(t + options['logEps'])

	if useTopics:
		return x, y, t, top, mask, lengths
	else:
		return x, y, t, mask, lengths

def padMatrixWithoutTime(seqs, labels, options, topics=None): # todo, add topics
	# # of visits for each patient
	lengths = np.array([len(seq) for seq in seqs]) - 1
	n_samples = len(seqs)
	maxlen = np.max(lengths)
	inputDimSize = options['inputDimSize']
	numClass = options['numClass']
	useTopics = options['useTopics']

	# visit-patient-code
	x = np.zeros((maxlen, n_samples, inputDimSize)).astype(config.floatX)
	y = np.zeros((maxlen, n_samples, numClass)).astype(config.floatX)
	# indicator of visits by first several elements
	mask = np.zeros((maxlen, n_samples)).astype(config.floatX)

	if useTopics:
		topicSize = options['topicSize']
		top = np.zeros((maxlen, n_samples, topicSize)).astype(config.floatX)
		for idx, topic in enumerate(topics):
			for tvec, tvalue in zip(top[:lengths[idx], idx,:], topic[:-1]):
				tvec = tvalue.copy()

	# transform from list to matrix
	for idx, (seq,label) in enumerate(zip(seqs,labels)):
		for xvec, subseq in zip(x[:,idx,:], seq[:-1]):
			xvec[subseq] = 1.
		for yvec, subseq in zip(y[:,idx,:], label[1:]):
			yvec[subseq] = 1.
		mask[:lengths[idx], idx] = 1.

	lengths = np.array(lengths, dtype=config.floatX)
	if useTopics:
		return x, y, top, mask, lengths
	else:
		return x, y, mask, lengths

def load_data(seqFile, labelFile, timeFile, topicFile):
	train_set_x = pickle.load(open(seqFile+'.train', 'rb'))
	valid_set_x = pickle.load(open(seqFile+'.valid', 'rb'))
	test_set_x = pickle.load(open(seqFile+'.test', 'rb'))
	train_set_y = pickle.load(open(labelFile+'.train', 'rb'))
	valid_set_y = pickle.load(open(labelFile+'.valid', 'rb'))
	test_set_y = pickle.load(open(labelFile+'.test', 'rb'))
	train_set_t = None
	valid_set_t = None
	test_set_t = None
	train_set_topic = None
	valid_set_topic = None
	test_set_topic = None

	if len(timeFile) > 0:
		train_set_t = pickle.load(open(timeFile+'.train', 'rb'))
		valid_set_t = pickle.load(open(timeFile+'.valid', 'rb'))
		test_set_t = pickle.load(open(timeFile+'.test', 'rb'))

	if len(topicFile) > 0:
		train_set_topic = pickle.load(open(topicFile+'.train', 'rb'))
		valid_set_topic = pickle.load(open(topicFile+'.valid', 'rb'))
		test_set_topic = pickle.load(open(topicFile+'.test', 'rb'))

	# sort patients in order of # visits
	def len_argsort(seq):
		return sorted(range(len(seq)), key=lambda x: len(seq[x]))

	train_sorted_index = len_argsort(train_set_x)
	train_set_x = [train_set_x[i] for i in train_sorted_index]
	train_set_y = [train_set_y[i] for i in train_sorted_index]

	valid_sorted_index = len_argsort(valid_set_x)
	valid_set_x = [valid_set_x[i] for i in valid_sorted_index]
	valid_set_y = [valid_set_y[i] for i in valid_sorted_index]

	test_sorted_index = len_argsort(test_set_x)
	test_set_x = [test_set_x[i] for i in test_sorted_index]
	test_set_y = [test_set_y[i] for i in test_sorted_index]

	if len(timeFile) > 0:
		train_set_t = [train_set_t[i] for i in train_sorted_index]
		valid_set_t = [valid_set_t[i] for i in valid_sorted_index]
		test_set_t = [test_set_t[i] for i in test_sorted_index]

	if len(topicFile) > 0:
		train_set_topic = [train_set_topic[i] for i in train_sorted_index]
		valid_set_topic = [valid_set_topic[i] for i in valid_sorted_index]
		test_set_topic = [test_set_topic[i] for i in test_sorted_index]

	train_set = (train_set_x, train_set_y, train_set_t,train_set_topic)
	valid_set = (valid_set_x, valid_set_y, valid_set_t,valid_set_topic)
	test_set = (test_set_x, test_set_y, test_set_t, test_set_topic)

	return train_set, valid_set, test_set

def calculate_auc(test_model, dataset, options):
	inputDimSize = options['inputDimSize']
	numClass = options['numClass']
	batchSize = options['batchSize']
	useTime = options['useTime']
	useTopics = options['useTopics']

	n_batches = int(np.ceil(float(len(dataset[0])) / float(batchSize)))
	aucSum = 0.0
	dataCount = 0.0
	for index in xrange(n_batches):
		batchX = dataset[0][index*batchSize:(index+1)*batchSize]
		batchY = dataset[1][index*batchSize:(index+1)*batchSize]
		if useTime and useTopics:
			batchT = dataset[2][index*batchSize:(index+1)*batchSize]
			batchTopics = dataset[3][index*batchSize:(index+1)*batchSize]
			x, y, t, topics, mask, lengths = padMatrixWithTime(batchX, batchY, batchT, options, batchTopics)
			auc = test_model(x, y, t, topics, mask, lengths)
		elif useTime:
			batchT = dataset[2][index*batchSize:(index+1)*batchSize]
			x, y, t, mask, lengths = padMatrixWithTime(batchX, batchY, batchT, options)
			auc = test_model(x, y, t, mask, lengths)
		elif useTopics:
			batchTopics = dataset[3][index*batchSize:(index+1)*batchSize]
			x, y, topics, mask, lengths = padMatrixWithoutTime(batchX, batchY, options, batchTopics)
			auc = test_model(x, y, topics, mask, lengths)
		else:
			x, y, mask, lengths = padMatrixWithoutTime(batchX, batchY, options)
			auc = test_model(x, y, mask, lengths)
		aucSum += auc * len(batchX)
		dataCount += float(len(batchX))
	return aucSum / dataCount

def train_doctorAI(
	seqFile='seqFile.txt',
	inputDimSize=20000,
	labelFile='labelFile.txt',
	numClass=500,
	outFile='outFile.txt',
	timeFile='timeFile.txt',
	tradeoff=1.0,
	useLogTime=True,
	embFile='embFile.txt',
	embSize=200,
	embFineTune=True,
	hiddenDimSize=[200,200],
	batchSize=100,
	max_epochs=20,
	L2_output=0.001,
	L2_time=0.001,
	dropout_rate=0.5,
	logEps=1e-8,
	verbose=False,
	topicFile='topicFile.txt',
	topicSize=20,
	experiment='1'
):
	options = locals().copy()

	if len(timeFile) > 0: useTime = True
	else: useTime = False
	options['useTime'] = useTime

	if len(topicFile) > 0: useTopics = True
	else: useTopics = False
	options['useTopics'] = useTopics

	print 'Initializing the parameters ... ',
	params = init_params(options)
	tparams = init_tparams(params, options)

	print 'Building the model ... ',
	f_grad_shared = None
	f_update = None
	if useTime and embFineTune:
		print 'using duration information, fine-tuning code representations'
		use_noise, x, y, t, topics, mask, lengths, cost =  build_model(tparams, options)
		grads = T.grad(cost, wrt=tparams.values())
		f_grad_shared, f_update = adadelta(tparams, grads, x, y, mask, lengths, cost, options, t, topics)
	elif useTime and not embFineTune:
		print 'using duration information, not fine-tuning code representations'
		W_emb = theano.shared(params['W_emb'], name='W_emb')
		use_noise, x, y, t, topics, mask, lengths, cost =  build_model(tparams, options, W_emb)
		grads = T.grad(cost, wrt=tparams.values())
		f_grad_shared, f_update = adadelta(tparams, grads, x, y, mask, lengths, cost, options, t, topics)
	elif not useTime and embFineTune:
		print 'not using duration information, fine-tuning code representations'
		use_noise, x, y, topics, mask, lengths, cost =  build_model(tparams, options)
		grads = T.grad(cost, wrt=tparams.values())
		f_grad_shared, f_update = adadelta(tparams, grads, x, y, mask, lengths, cost, options, None, topics)
	elif not useTime and not embFineTune:
		print 'not using duration information, not fine-tuning code representations'
		W_emb = theano.shared(params['W_emb'], name='W_emb')
		use_noise, x, y, topics, mask, lengths, cost =  build_model(tparams, options, W_emb)
		grads = T.grad(cost, wrt=tparams.values())
		f_grad_shared, f_update = adadelta(tparams, grads, x, y, mask, lengths, cost, options, None, topics)

	print 'Loading data ... ',
	trainSet, validSet, testSet = load_data(seqFile, labelFile, timeFile,topicFile)
	n_batches = int(np.ceil(float(len(trainSet[0])) / float(batchSize)))
	print 'done'

	if useTime and useTopics:
		test_model = theano.function(inputs=[x, y, t, topics, mask, lengths], outputs=cost, name='test_model')
	elif useTime:
		test_model = theano.function(inputs=[x, y, t, mask, lengths], outputs=cost, name='test_model')
	elif useTopics:
		test_model = theano.function(inputs=[x, y, topics, mask, lengths], outputs=cost, name='test_model') # todo
	else:
		test_model = theano.function(inputs=[x, y, mask, lengths], outputs=cost, name='test_model')

	bestValidCrossEntropy = 1e20
	bestValidEpoch = 0
	testCrossEntropy = 0.0
	print 'Optimization start !!',max_epochs
	for epoch in xrange(max_epochs):
		iteration = 0
		costVector = []
		for index in random.sample(range(n_batches), n_batches):
			use_noise.set_value(1.)
			batchX = trainSet[0][index*batchSize:(index+1)*batchSize]
			batchY = trainSet[1][index*batchSize:(index+1)*batchSize]
			if useTime and useTopics:
				batchT = trainSet[2][index*batchSize:(index+1)*batchSize]
				batchTopics = trainSet[3][index*batchSize:(index+1)*batchSize]
				x, y, t, topics, mask, lengths = padMatrixWithTime(batchX, batchY, batchT, options, batchTopics)
				cost = f_grad_shared(x, y, t, topics, mask, lengths)
			elif useTime:
				batchT = trainSet[2][index*batchSize:(index+1)*batchSize]
				x, y, t, mask, lengths = padMatrixWithTime(batchX, batchY, batchT, options)
				cost = f_grad_shared(x, y, t, mask, lengths)
			elif useTopics:
				batchTopics = trainSet[3][index*batchSize:(index+1)*batchSize]
				x, y, topics, mask, lengths = padMatrixWithoutTime(batchX, batchY, options,batchTopics)
				cost = f_grad_shared(x, y, topics, mask, lengths)
			else:
				x, y, mask, lengths = padMatrixWithoutTime(batchX, batchY, options)
				cost = f_grad_shared(x, y, mask, lengths)
			costVector.append(cost)
			f_update()
			if (iteration % 10 == 0) and verbose: print 'epoch:%d, iteration:%d/%d, cost:%f' % (epoch, iteration, n_batches, cost)
			iteration += 1

		print 'epoch:%d, mean_cost:%f' % (epoch, np.mean(costVector))
		use_noise.set_value(0.)
		validAuc = calculate_auc(test_model, validSet, options)
		print 'Validation cross entropy:%f at epoch:%d' % (validAuc, epoch)
		if validAuc < bestValidCrossEntropy:
			bestValidCrossEntropy = validAuc
			bestValidEpoch = epoch
			bestParams = unzip(tparams)
			testCrossEntropy = calculate_auc(test_model, testSet, options)
			print 'Test cross entropy:%f at epoch:%d' % (testCrossEntropy, epoch)
			tempParams = unzip(tparams)
			np.savez_compressed(outFile + 'model_exp' + str(experiment) + '.' + str(epoch), **tempParams)
	print 'The best valid cross entropy:%f at epoch:%d' % (bestValidCrossEntropy, bestValidEpoch)
	print 'The test cross entropy: %f' % testCrossEntropy

def parse_arguments(parser):
	parser.add_argument('seq_file', type=str, metavar='<visit_file>', help='The path to the Pickled file containing visit information of patients')
	parser.add_argument('n_input_codes', type=int, metavar='<n_input_codes>', help='The number of unique input medical codes')
	parser.add_argument('label_file', type=str, metavar='<label_file>', help='The path to the Pickled file containing label information of patients')
	parser.add_argument('n_output_codes', type=int, metavar='<n_output_codes>', help='The number of unique label medical codes')
	parser.add_argument('out_file', metavar='out_file', help='The path to the output models. The models will be saved after every epoch')
	parser.add_argument('--time_file', type=str, default='', help='The path to the Pickled file containing durations between visits of patients. If you are not using duration information, do not use this option')
	parser.add_argument('--predict_time', type=int, default=0, choices=[0,1], help='Use this option if you want the GRU to also predict the time duration until the next visit (0 for false, 1 for true) (default value: 0)')
	parser.add_argument('--tradeoff', type=float, default=1.0, help='Tradeoff variable for balancing the two loss functions: code prediction function and duration prediction function (default value: 1.0)')
	parser.add_argument('--use_log_time', type=int, default=1, choices=[0,1], help='Use logarithm of time duration to dampen the impact of the outliers (0 for false, 1 for true) (default value: 1)')
	parser.add_argument('--embed_file', type=str, default='', help='The path to the Pickled file containing the representation vectors of medical codes. If you are not using medical code representations, do not use this option')
	parser.add_argument('--embed_size', type=int, default=200, help='The size of the visit embedding before passing it to the GRU layers. If you are not providing your own medical code vectors, you must specify this value (default value: 200)')
	parser.add_argument('--embed_finetune', type=int, default=1, choices=[0,1], help='If you are using randomly initialized code representations, always use this option. If you are using an external medical code representations, and you want to fine-tune them as you train the GRU, use this option as well. (0 for false, 1 for true) (default value: 1)')
	parser.add_argument('--hidden_dim_size', type=str, default='[200,200]', help='The size of the hidden layers of the GRU. This is a string argument. For example, [500,400] means you are using a two-layer GRU where the lower layer uses a 500-dimensional hidden layer, and the upper layer uses a 400-dimensional hidden layer. (default value: [200,200])')
	parser.add_argument('--batch_size', type=int, default=100, help='The size of a single mini-batch (default value: 100)')
	parser.add_argument('--n_epochs', type=int, default=100, help='The number of training epochs (default value: 10)')
	parser.add_argument('--L2_softmax', type=float, default=0.001, help='L2 regularization for the softmax function (default value: 0.001)')
	parser.add_argument('--L2_time', type=float, default=0.001, help='L2 regularization for the linear regression (default value: 0.001)')
	parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate between GRU hidden layers, and between the final hidden layer and the softmax layer (default value: 0.5)')
	parser.add_argument('--log_eps', type=float, default=1e-8, help='A small value to prevent log(0) (default value: 1e-8)')
	parser.add_argument('--verbose', action='store_true', help='Print output after every 10 mini-batches (default false)')

	parser.add_argument('--topic_file', type=str, default='', help='The path to the Pickled file containing topics covered in each visit. If you are not using topics information, do not use this option')
	parser.add_argument('--topic_size', type=int, default=20, help='The size of the topics. (default value: 20)')
	parser.add_argument('--experiment', type=str, default='1', help='The order of experiment. (default value: 1)')

	args = parser.parse_args()
	return args

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	args = parse_arguments(parser)
	hiddenDimSize = [int(strDim) for strDim in args.hidden_dim_size[1:-1].split(',')]
	print 'hiddenDimSize:' + str(hiddenDimSize)

	if args.predict_time and args.time_file == '':
		print 'Cannot predict time duration without time file'
		sys.exit()

	train_doctorAI(
		seqFile=args.seq_file,
		inputDimSize=args.n_input_codes,
		labelFile=args.label_file,
		numClass=args.n_output_codes,
		outFile=args.out_file,
		timeFile=args.time_file,
		tradeoff=args.tradeoff,
		useLogTime=args.use_log_time,
		embFile=args.embed_file,
		embSize=args.embed_size,
		embFineTune=args.embed_finetune,
		hiddenDimSize=hiddenDimSize,
		batchSize=args.batch_size,
		max_epochs=args.n_epochs,
		L2_output=args.L2_softmax,
		L2_time=args.L2_time,
		dropout_rate=args.dropout_rate,
		logEps=args.log_eps,
		verbose=args.verbose,

		topicFile=args.topic_file,
		topicSize=args.topic_size,
		experiment = args.experiment
	)
