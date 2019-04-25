from PIL import Image
import os
import pdb
fs=os.listdir('personalized')
for f in fs:
	img=Image.open('personalized/'+f).convert('L')
	#pdb.set_trace()
	img=img.resize((128,128))
	t=f.replace('png','jpg')
	img.save('personalized/'+t)