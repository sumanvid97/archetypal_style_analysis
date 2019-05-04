import os
import re
import torch
import argparse
import torchvision
# import autoencoder
from log_utils import get_logger
from torch.utils.data import DataLoader
# from torchsummary import summary
from timeit import default_timer as timer
# from sklearn.decomposition import PCA

import Datasets
import encoder_decoder_factory

log = get_logger()

def parse_args():
	parser = argparse.ArgumentParser(description='Pytorch implementation of arbitrary style transfer via CNN features WCT trasform',
									 epilog='Supported image file formats are: jpg, jpeg, png')

	parser.add_argument('--content', help='Path of the content image (or a directory containing images) to be trasformed')
	parser.add_argument('--style', help='Path of the style image (or a directory containing images) to use')
	parser.add_argument('--synthesis', default=False, action='store_true', help='Flag to syntesize a new texture. Must provide a texture style image')
	parser.add_argument('--stylePair', help='Path of two style images (separated by ",") to use in combination')
	parser.add_argument('--mask', help='Path of the binary mask image (white on black) to trasfer the style pair in the corrisponding areas')

	parser.add_argument('--contentSize', type=int, help='Reshape content image to have the new specified maximum size (keeping aspect ratio)') # default=768 in the paper
	parser.add_argument('--styleSize', type=int, help='Reshape style image to have the new specified maximum size (keeping aspect ratio)')

	parser.add_argument('--dataDir', default='../Van.gogh.paintings/', help='Path of the directory where stylized results will be saved')
	parser.add_argument('--outDir', default='outputs', help='Path of the directory where stylized results will be saved')
	parser.add_argument('--outPrefix', help='Name prefixed in the saved stylized images')

	parser.add_argument('--alpha', type=float, default=0.2, help='Hyperparameter balancing the blending between original content features and WCT-transformed features')
	parser.add_argument('--beta', type=float, default=0.5, help='Hyperparameter balancing the interpolation between the two images in the stylePair')
	parser.add_argument('--no-cuda', default=False, action='store_true', help='Flag to enables GPU (CUDA) accelerated computations')
	parser.add_argument('--single-level', default=False, action='store_true', help='Flag to switch to single level stylization')
	args = parser.parse_args(['--content','inputs/contents/face.jpg', '--style','inputs/styles/in4.jpg', '--contentSize','768','--styleSize','256'])
	return args

def feature_extract(level, content, encoders, svd_device, cnn_device):
	with torch.no_grad():
		cf = encoders[level](content).data.to(device=svd_device).squeeze(0)
	return cf

def save_image(img, content_name, style_name, out_ext, args):
	torchvision.utils.save_image(img.cpu().detach().squeeze(0),
	 os.path.join(args.outDir,
	  (args.outPrefix + '_' if args.outPrefix else '') + content_name + '_stylized_by_' + style_name + '_alpha_' + str(int(args.alpha*100)) + '.' + out_ext))

def main():
	start = timer()

	args = parse_args()

	try:
		os.makedirs(args.outDir, exist_ok=True)
	except OSError:
		log.exception('Error encoutered while creating output directory ' + args.outDir)

	if not args.no_cuda and torch.cuda.is_available():
		log.info('Utilizing the first CUDA gpu available')
		args.device = torch.device('cuda:0')
	else:
		log.info('Utilizing the cpu for computations')
		args.device = torch.device('cpu')

	dataset = Datasets.PaintingsDataset(args)
	dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
	
	# dim = dataset[0]['content'].size()
	
	# model1 = encoder_decoder_factory.Encoder(1)
	# model1.to(device=args.device)
	# summary(model1,dim)
	# model2 = encoder_decoder_factory.Encoder(2)
	# model2.to(device=args.device)
	# summary(model2,dim)
	# model3 = encoder_decoder_factory.Encoder(3)
	# model3.to(device=args.device)
	# summary(model3,dim)
	# model4 = encoder_decoder_factory.Encoder(4)
	# model4.to(device=args.device)
	# summary(model4,dim)
	# modelL = encoder_decoder_factory.Encoder(5)
	# modelL.to(device=args.device)
	# summary(modelL,dim)
	
	encoders = []
	L = 4
	N = len(dataloader)
	# D = 611776	#for L=5
	# D = 349120	#for L=4
	# D = 86464		#for L=3
	# D = 20672		#for L=2
	# D = 4160		#for L=1
	for l in range(L):
		encoders.append(encoder_decoder_factory.Encoder(l+1).to(device=args.device))
	
	descriptors = torch.empty(0)
	# means = [torch.empty(0)]*L
	# sigmas = [torch.empty(0)]*L
	for i, sample in enumerate(dataloader):
		log.info(str(i+1) + '/'+ str(len(dataloader)) + ' computing style descriptor for ' + str(sample['name']))
		content = sample['content'].to(device=args.device)
		
		style_desc_flat = torch.empty(0)
		for l in range(L):
			# print('layer',l+1)
			tens = feature_extract(l, content, encoders, torch.device('cpu'), torch.device('cuda:0'))
			# print('tensor',tens.size())
			feature = tens.reshape(tens.size(0),-1)
			# print('feature',feature.size())
			P = feature.size(0)
			M = feature.size(1)
			mew = torch.sum(feature, dim=1)
			mew = mew/(M*P*(P+1))
			# print('mean',mew.size())
			sigma = torch.zeros([P,P])
			U = feature.sub(mew[:,None])
			sigma = torch.mm(U,torch.t(U))
			sigma = sigma/(M*P*(P+1))
			# if i==0:
			# 	means[l] = mew.view(1,P)
			# 	sigmas[l] = sigma.view(1,P,P)
			# else:
			# 	means[l] = torch.cat([means[l],mew.view(1,P)])
			# 	sigmas[l] = torch.cat([sigmas[l],sigma.view(1,P,P)])	   
			# print('sigma',sigma.size())
			style_desc = torch.cat([mew,sigma.resize_(P*P)])
			# print('descriptor',style_desc.size())
			style_desc_flat = torch.cat([style_desc_flat,style_desc])
			# print('flat descriptor',style_desc_flat.size())
			# print(means[l].size())
			# print(sigmas[l].size())
		style_desc_flat = style_desc_flat.view(1,-1)
		if i==0:
			descriptors = style_desc_flat
		else:
			descriptors = torch.cat([descriptors,style_desc_flat])
		# print('descriptors',descriptors.size())

	# for l in range(L):
	# 	torch.save(descriptors, 'tensors/means_'+str(L)+'_'+str(l)+'.pt')
	# 	torch.save(descriptors, 'tensors/sigmas_'+str(L)+'_'+str(l)+'.pt')	

	# k = 32
	p = 512
	torch.save(descriptors, 'tensors/descriptor'+str(L)+'.pt')
	U,S,V = torch.svd(descriptors)
	print(descriptors.size())
	print(U.size())
	print(S.size())
	# print(S)
	print(V.size())
	X = torch.mm(descriptors,V[:,:p])
	torch.save(X, 'tensors/tensor'+str(L)+'.pt')
	# print(X)
	descriptors_var = 0
	for i in range(N):
		descriptors_var += S[i]**2
	descriptors_var /= N
	X_var = 0
	for i in range(p):
		X_var += S[i]**2
	X_var /= N
	log.info('Original variance was ' + str(descriptors_var))
	log.info('New variance is ' + str(X_var))
	log.info('Retention is ' + str(X_var/descriptors_var))	

	# pca = PCA(n_components=p)
	# pca.fit(descriptors)
	# Y = pca.transform(descriptors)
	# print(Y)
	# V = pca.singular_values_
	# var = 0
	# for i in range(len(V)):
	# 	var += V[i]**2
	# var /= N	
	# log.info('New variance is ' + str(var))
	# log.info('Retaintion is ' + str(var/descriptors_var))	

	end = timer()
	log.info('Time taken for dimensionality reduction: ' + str(end - start) + 's')
				

if __name__ == "__main__":
	main()