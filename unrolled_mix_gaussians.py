# Most of this file comes from git@github.com:poolio/unrolled_gan.git
from collections import OrderedDict
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import scipy as sp
import scipy.stats
import time
import os
from tqdm import tqdm

dirname = os.path.dirname(__file__)

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
    unrolling_steps=4,  # CHANGE THIS TO 0 TO REPRODUCE THE EXPERIMENTS FOR 'UNROLLED 0'
)

param_string = 'unrolled_'+str(params['unrolling_steps'])+'_disclr_'+str(params['disc_learning_rate'])+\
               '_gen_lr'+str(params['gen_learning_rate'])

generate_movie = False
_graph_replace = tf.contrib.graph_editor.graph_replace


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
    # ax.set_title('Step '+str(int(i))+'k ('+str(int(time))+'s)')
    ax.set_title(str(int(t)) + 's')
    ax.axis('off')


def remove_original_op_attributes(graph):
    """Remove _original_op attribute from all operations in a graph."""
    for op in graph.get_operations():
        op._original_op = None


def graph_replace(*args, **kwargs):
    """Monkey patch graph_replace so that it works with TF 1.0"""
    remove_original_op_attributes(tf.get_default_graph())
    return _graph_replace(*args, **kwargs)


def extract_update_dict(update_ops):
    """Extract variables and their new values from Assign and AssignAdd ops.

    Args:
        update_ops: list of Assign and AssignAdd ops, typically computed using Keras' opt.get_updates()

    Returns:
        dict mapping from variable values to their updated value
    """
    name_to_var = {v.name: v for v in tf.global_variables()}
    updates = OrderedDict()
    for update in update_ops:
        var_name = update.op.inputs[0].name
        var = name_to_var[var_name]
        value = update.op.inputs[1]
        if update.op.type == 'Assign':
            updates[var.value()] = value
        elif update.op.type == 'AssignAdd':
            updates[var.value()] = var + value
        else:
            raise ValueError("Update op type (%s) must be of type Assign or AssignAdd" % update_ops.op.type)
    return updates


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
d_train_op = tf.group(*updates, name="d_train_op")

# Unroll optimization of the discrimiantor
if params['unrolling_steps'] > 0:
    # Get dictionary mapping from variables to their update value after one optimization step
    update_dict = extract_update_dict(updates)
    cur_update_dict = update_dict
    for i in range(params['unrolling_steps'] - 1):
        # Compute variable updates given the previous iteration's updated variable
        cur_update_dict = graph_replace(update_dict, cur_update_dict)
    # Final unrolled loss uses the parameters at the last time step
    unrolled_loss = graph_replace(loss, cur_update_dict)
else:
    unrolled_loss = loss

# Optimize the generator on the unrolled loss
g_train_opt = tf.train.AdamOptimizer(params['gen_learning_rate'], beta1=params['beta1'], epsilon=params['epsilon'])
g_train_op = g_train_opt.minimize(-unrolled_loss, var_list=gen_vars)

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


def visualize():
    np_samples.append(np.vstack([sess.run(samples) for _ in range(n_batches_viz)]))
    xx, yy = sess.run([samples, data])
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(xx[:, 0], xx[:, 1], edgecolor='none')
    plt.scatter(yy[:, 0], yy[:, 1], c='g', edgecolor='none')
    plt.axis('off')
    plt.show()


def save_progress(iter):

    np_samples.append(np.vstack([sess.run(samples) for _ in range(n_batches_viz)]))
    xx, yy = sess.run([samples, data])
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(xx[:, 0], xx[:, 1], edgecolor='none')
    plt.scatter(yy[:, 0], yy[:, 1], c='g', edgecolor='none')
    plt.axis('off')

    fig.suptitle('Generator at iteration'+str(iter))  # or plt.suptitle('Main title')
    plt.savefig(dirname+'/gauss_mix_unrolled_4_images/'+'iteration'+str(iter)+'.png')


start = time.time()
for i in tqdm(range(params['max_iter'])):
    f, _, _ = sess.run([[loss, unrolled_loss], g_train_op, d_train_op])
    fs.append(f)
    if i % viz_every == 0:
        save_progress(i)
        time_stamps.append(int(time.time() - start))

plt.close()

# plot kdes
num_plots = len(np_samples)
bbox = [-4, 4, -4, 4]
fig, list_ax = plt.subplots(1, num_plots)

for p in range(num_plots):
    kde(np_samples[p][0:1000, 0], np_samples[p][0:1000, 1], i=int(p*params['viz_every']/1000), t=time_stamps[p],
        bbox=bbox, ax=list_ax[p])

plt.savefig(dirname + '/gauss_mix_unrolled_4_images/' + 'kde_'+param_string+'.eps')
plt.close()
