# implementation of SP-FTL on mixture of gaussians
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import os
import scipy as sp
from tqdm import tqdm

dirname = os.path.dirname(os.path.abspath(__file__))

ds = tf.contrib.distributions
slim = tf.contrib.slim

params = dict(
    batch_size=512,
    disc_learning_rate=1e-4,
    gen_learning_rate=1e-4,
    beta1=0.5,
    epsilon=1e-8,
    max_iter=20001,
    viz_every=4000,
    z_dim=256,
    x_dim=2,
    sp_ftl_iters=1000,  # CONTROLS HOW OFTEN WE AVERAGE THE ITERATES
    average_gen=True,
    average_disc=True
)

param_string = '_disclr_'+str(params['disc_learning_rate'])+'_gen_lr'+str(params['gen_learning_rate'])


def kde(mu, tau, i, t=None, bbox=None, ax=None, xlabel="", ylabel="", cmap='Blues'):
    """Creates an estimate of the density of the generator"""
    values = np.vstack([mu, tau])
    kernel = sp.stats.gaussian_kde(values)

    ax.axis(bbox)
    ax.set_aspect(abs(bbox[1] - bbox[0]) / abs(bbox[3] - bbox[2]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    xx, yy = np.mgrid[bbox[0]:bbox[1]:300j, bbox[2]:bbox[3]:300j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kernel(positions).T, xx.shape)
    ax.contourf(xx, yy, f, cmap=cmap)
    ax.set_title(str(int(t)) + 's')
    ax.axis('off')


def sample_mog(batch_size, n_mixture=8, std=0.02, radius=2.0):
    thetas = np.linspace(0, 2 * np.pi, n_mixture)
    xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
    cat = ds.Categorical(tf.zeros(n_mixture))
    comps = [ds.MultivariateNormalDiag([xi, yi], [std, std]) for xi, yi in zip(xs.ravel(), ys.ravel())]
    data = ds.Mixture(cat, comps)
    return data.sample(batch_size)


def generator(z, output_dim=2, n_hidden=128, n_layer=2):
    with tf.variable_scope("generator"):
        h = slim.stack(z, slim.fully_connected, [n_hidden] * n_layer, activation_fn=tf.nn.relu)
        x = slim.fully_connected(h, output_dim, activation_fn=None)
    return x


def discriminator(x, n_hidden=128, n_layer=1, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        # scale down by a factor of 4
        x = tf.scalar_mul(1.0/4.0, x)

        # one layer relu fully connected
        h = slim.stack(x, slim.fully_connected, [n_hidden] * n_layer, activation_fn=tf.nn.relu)
        log_d = slim.fully_connected(h, 1, activation_fn=None)  # this layer does nothing
    return log_d


tf.reset_default_graph()

data = sample_mog(params['batch_size'])

noise = ds.Normal(tf.zeros(params['z_dim']),
                  tf.ones(params['z_dim'])).sample(params['batch_size'])

# Construct generator and discriminator nets
with slim.arg_scope([slim.fully_connected], weights_initializer=tf.orthogonal_initializer(gain=.8)):
    samples = generator(noise, output_dim=params['x_dim'])
    real_score = discriminator(data)
    fake_score = discriminator(samples, reuse=True)

# Saddle objective
loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=real_score, labels=tf.ones_like(real_score)) +
    tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_score, labels=tf.zeros_like(fake_score)))

gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")

# Vanilla discriminator update
d_opt = Adam(lr=params['disc_learning_rate'], beta_1=params['beta1'], epsilon=params['epsilon'])
updates = d_opt.get_updates(disc_vars, [], loss)
print('updates!', updates)

d_train_op = tf.group(*updates, name="d_train_op")

# Optimize the generator
g_train_opt = tf.train.AdamOptimizer(params['gen_learning_rate'], beta1=params['beta1'], epsilon=params['epsilon'])
g_train_op = g_train_opt.minimize(-loss, var_list=gen_vars)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# visualization parameters
xmax = 3
fs = []
frames = []
np_samples = []
time_stamps = []
n_batches_viz = 10
viz_every = params['viz_every']


def print_gen_vars():
    print(sess.run(gen_vars)[0][0][0:5])


# copy both nets to keep track of the sum of networks
gen_vars_copy = []
for g in range(len(gen_vars)):
    gen_vars_copy.append(tf.get_variable(name='gen_vars_copy_'+str(g), initializer=sess.run(gen_vars[g])))
    sess.run(gen_vars_copy[g].initializer)

disc_vars_copy = []
for d in range(len(disc_vars)):
    disc_vars_copy.append(tf.get_variable(name='disc_vars_copy'+str(d), initializer=sess.run(disc_vars[d])))
    sess.run(disc_vars_copy[d].initializer)

# create sum operations
list_sum_gen = [gen_vars_copy[g].assign_add(gen_vars[g]) for g in range(len(gen_vars))]
sum_gen_op = tf.group(*list_sum_gen, name='sum_consec_gen')

list_sum_disc = [disc_vars_copy[d].assign_add(disc_vars[d]) for d in range(len(disc_vars))]
sum_disc_op = tf.group(*list_sum_disc, name='sum_consec_disc')


def save_progress(iter):

    np_samples.append(np.vstack([sess.run(samples) for _ in range(n_batches_viz)]))
    xx, yy = sess.run([samples, data])
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(xx[:, 0], xx[:, 1], edgecolor='none')
    plt.scatter(yy[:, 0], yy[:, 1], c='g', edgecolor='none')
    plt.axis('off')

    fig.suptitle('Generator at iteration'+str(iter))  # or plt.suptitle('Main title')
    plt.savefig(dirname+'/gauss_mix_spftl_images/'+'iteration'+str(iter)+'.png')


# Training loop
for i in tqdm(range(params['max_iter'])):

    f, _, _ = sess.run([[loss, loss], g_train_op, d_train_op])

    # add new weights to the copies
    if params['average_gen']:
        sess.run(sum_gen_op)
    if params['average_disc']:
        sess.run(sum_disc_op)

    if i % params['sp_ftl_iters'] == 0 and i != 0:
        if params['average_gen']:
            # copy = copy * 1/sp_ftl_iters
            # generator
            sess.run([gen_vars_copy[g].assign(tf.scalar_mul(1.0 / (params['sp_ftl_iters'] + 2.0), gen_vars_copy[g]))
                      for g in range(len(gen_vars_copy))])
            # set the nets to be equal to copy
            # generator
            sess.run([gen_vars[g].assign(gen_vars_copy[g]) for g in range(len(gen_vars_copy))])
        if params['average_disc']:
            # copy = copy * 1/sp_ftl_iters
            # discriminator
            sess.run([disc_vars_copy[d].assign(tf.scalar_mul(1.0/(params['sp_ftl_iters']+2.0), disc_vars_copy[d]))
                      for d in range(len(disc_vars_copy))])

            # discriminator
            sess.run([disc_vars[d].assign(disc_vars_copy[d]) for d in range(len(disc_vars_copy))])


    fs.append(f)
    if i % viz_every == 0:
        save_progress(i)
        pass

plt.show()

# plot kdes
num_plots = len(np_samples)
bbox = [-4, 4, -4, 4]
fig, list_ax = plt.subplots(1, num_plots)

for p in range(num_plots):
    kde(np_samples[p][0:1000, 0], np_samples[p][0:1000, 1], i=int(p*params['viz_every']/1000), t=time_stamps[p],
        bbox=bbox, ax=list_ax[p])

plt.savefig(dirname + '/gauss_mix_unrolled_4_images/' + 'kde_'+param_string+'.eps')
plt.close()