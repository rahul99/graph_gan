from keras.optimizers import Adam

def init():
	global cond_data # condition data
	global trgt_data # target data
	global gen_inps_pair # generator input pair (condition, target graph)
	global dis_inps_pair # discriminator input pair (condition, target data)
	global myGenerator # generator model object
	global myDiscriminator # discriminator model object
	global CGAN # stacked conditional GAN model object
	global losses # loss list for generator and discriminator
	
	global gen_opt # generator optimizer
	gen_opt = Adam(lr=1e-3) # starting (default) value

	global dis_opt # discriminaotor optimizer
	dis_opt = Adam(lr=1e-2) # starting (default) value

