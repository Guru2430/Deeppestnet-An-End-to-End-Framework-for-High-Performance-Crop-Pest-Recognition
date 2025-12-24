
# from model import 


from utils import LoadData
import tensorflow as tf
from model import Proposed

import optimizer



def train(width_per_group, bottleneck_ratio, num_blocks, units,
          shuffle_buffer_size=100, batch_size=8, epochs=300):
    
    load_data = LoadData()
    train_data, train_labels = load_data.load(name='train_set')
    
    print("training data size", train_data.data_size)
    
    print("number of train images", len(train_data))
    
    model = Proposed(width_per_group, bottleneck_ratio, num_blocks, units)   
    
    
    optimizer_fun = tf.keras.optimizers.SGD(learning_rate=0.01)
    
    loss_fn = tf.keras.losses.Crossentropy(from_logits=True)
    
    
    init = tf.global_variables_initializer()
    

    
    with tf.Session() as sess:
        sess.run(init)
        for e in range(1, epochs + 1):


            train_data.reset_index()
            i = 0
        

            clf_features = []
        
            while train_data.is_available():
                i += 1
        
                batch_x1, batch_x2, batch_y, batch_features = train_data.next_batch(batch_size=batch_size)
        
                merged, _, c, features = sess.run([model.merged, optimizer, model.cost, model.output_features],
                                                  feed_dict={model.x1: batch_x1,
                                                             model.x2: batch_x2,
                                                             model.y: batch_y,
                                                             model.features: batch_features})
        
                clf_features.append(features)
            
                           
                with tf.GradientTape() as tape:
                    logits = model(batch_x1, training=True)
                    loss_value = loss_fn(batch_y, logits)
                    
                optimizer.run()    
                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer_fun.apply_gradients(zip(grads, model.trainable_weights))
            
    
    model.save('model/model.h5')
    



