import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from tqdm import tqdm
import Mean
from util import sort_corr, init_attn_2, categorical_to_code, Data_convert, get_M_by_data_m, get_number_data_mu_var, \
    reconvert_data
import torch


def sample_batch_index(total, batch_size):
    total_idx = np.random.permutation(total)
    batch_idx = total_idx[:batch_size]
    return batch_idx
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)
def gain(data_m, data_x, con_cols, gain_parameters,value_cat, enc, values, device):
    # System parameters
    batch_size = gain_parameters['batch_size']
    hint_rate = gain_parameters['hint_rate']
    alpha = gain_parameters['alpha']
    iterations = gain_parameters['iterations']
    # Other parameters
    no, data_dim = data_x.shape
    attention_impute_data = pd.DataFrame(data_x)
    attention_impute_data.columns = values
    cat_to_code_data, enc = categorical_to_code(attention_impute_data.copy(), value_cat, enc)
    cat_to_code_data.columns = [x for x in range(cat_to_code_data.shape[1])]
    fields, feed_data = Data_convert(cat_to_code_data, "minmax", con_cols)

    M_tensor = get_M_by_data_m(data_m, fields, device)
    M_numpy = M_tensor.cpu().numpy()
    impute_data_code = torch.tensor(feed_data.values, dtype=torch.float).to(device)
    zero_feed_data_code = impute_data_code * M_tensor
    _, code_dim = zero_feed_data_code.shape
    X_numpy = impute_data_code.cpu().numpy()
    # Hidden state dimensions
    dim = int(code_dim)
    h_dim = int(code_dim)

    ## GAIN architecture
    # Input placeholders
    # Data vector
    X = tf.placeholder(tf.float32, shape=[None, dim])
    # Mask vector
    M = tf.placeholder(tf.float32, shape=[None, dim])
    Data_M = tf.placeholder(tf.float32, shape=[None, data_dim])
    # Hint vector
    H = tf.placeholder(tf.float32, shape=[None, dim])

    # Discriminator variables
    D_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))  # Data + Hint as inputs
    D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    D_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

    D_W3 = tf.Variable(xavier_init([h_dim, data_dim]))
    D_b3 = tf.Variable(tf.zeros(shape=[data_dim]))  # Multi-variate outputs

    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

    # Generator variables
    # Data + Mask as inputs (Random noise is in missing components)
    G_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W3 = tf.Variable(xavier_init([h_dim, dim]))
    G_b3 = tf.Variable(tf.zeros(shape=[dim]))

    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]
    ## GAIN functions
    # Generator
    def generator(x, m):
        # Concatenate Mask and Data
        inputs = tf.concat(values=[x, m], axis=1)
        G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)

        G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
        return G_prob
    # Discriminator
    def discriminator(x, h):
        # Concatenate Data and Hint
        inputs = tf.concat(values=[x, h], axis=1)
        D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        D_logit = tf.matmul(D_h2, D_W3) + D_b3
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob
    ## GAIN structure
    # Generator
    G_sample = generator(X, M)
    # Combine with observed data
    Hat_X = X * M + G_sample * (1 - M)
    # Discriminator
    D_prob = discriminator(Hat_X, H)
    ## GAIN loss
    D_loss_temp = -tf.reduce_mean(Data_M * tf.log(D_prob + 1e-8) \
                                  + (1 - Data_M) * tf.log(1. - D_prob + 1e-8))
    G_loss_temp = -tf.reduce_mean((1 - Data_M) * tf.log(D_prob + 1e-8))
    MSE_loss = \
        tf.reduce_mean((M * X - M * G_sample) ** 2) / tf.reduce_mean(M)
    D_loss = D_loss_temp
    G_loss = G_loss_temp + alpha * MSE_loss
    ## GAIN solver
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
    ## Iterations
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # Start Iterations
    for it in tqdm(range(iterations)):
        # Sample batch
        batch_idx = sample_batch_index(no, batch_size)
        X_mb = X_numpy[batch_idx, :]
        M_mb = M_numpy[batch_idx, :]
        data_M_mb = data_m[batch_idx, :]
        # Sample random vectors
        Z_mb = uniform_sampler(0, 0.01, batch_size, dim)
        # Sample hint vectors
        H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
        H_mb = M_mb * H_mb_temp

        # Combine random vectors with observed vectors
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

        _, D_loss_curr = sess.run([D_solver, D_loss_temp],
                                  feed_dict={Data_M: data_M_mb, X: X_mb, H: H_mb, M: M_mb})
        _, G_loss_curr, MSE_loss_curr = \
            sess.run([G_solver, G_loss_temp, MSE_loss],
                     feed_dict={X: X_mb, Data_M: data_M_mb, H: H_mb, M: M_mb})
    Z_mb = uniform_sampler(0, 0.01, no, dim)
    M_mb = M_numpy
    X_mb = X_numpy
    X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
    imputed_data = sess.run([G_sample], feed_dict={X: X_mb, M: M_mb})[0]
    imputed_data_code = M_numpy * X_numpy + (1 - M_numpy) * imputed_data
    code = torch.tensor(imputed_data_code)
    impute_data = reconvert_data(code, fields, value_cat, values, attention_impute_data.copy(), data_m, enc)
    return impute_data.values


def get_GAIN_filled(data_m, nan_data, con_cols, cat_cols,value_cat, enc, values, device):
    gain_parameters = {'batch_size':128,
                       'hint_rate': 0.5,
                       'alpha': 100,
                       'iterations': 1000}
    fill_mean_data = Mean.fill_data_mean(nan_data, con_cols)
    filled_numpy = gain(data_m, fill_mean_data, con_cols, gain_parameters,value_cat, enc, values, device)
    return filled_numpy


def uniform_sampler(low, high, rows, cols):
    '''Sample uniform random variables.

    Args:
      - low: low limit
      - high: high limit
      - rows: the number of rows
      - cols: the number of columns

    Returns:
      - uniform_random_matrix: generated uniform random matrix.
    '''
    return np.random.uniform(low, high, size=[rows, cols])


def binary_sampler(p, rows, cols):
    '''Sample binary random variables.

    Args:
      - p: probability of 1
      - rows: the number of rows
      - cols: the number of columns

    Returns:
      - binary_random_matrix: generated binary random matrix.
    '''
    unif_random_matrix = np.random.uniform(0., 1., size=[rows, cols])
    binary_random_matrix = 1 * (unif_random_matrix < p)
    return binary_random_matrix

