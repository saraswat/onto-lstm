import sys
from overrides import overrides
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.layers import InputSpec
from tensorflow.keras.layers import Layer, LSTM, Dense

def TF_PRINT(x, s, expected_shape=None, first_n=1):
    x = tf.convert_to_tensor(x)
    return tf.Print(x, [tf.shape(x)] if expected_shape==None else [tf.shape(x),
                                                                   expected_shape
#                                                                   ,tf.equal(tf.shape(x),expected_shape)
                                                                  ,tf.reduce_prod(tf.cast(tf.equal(tf.shape(x),expected_shape),tf.int32))
                                                                   ],
                    s, first_n=first_n, name=s)

class NSE(Layer):
    '''
    Simple Neural Semantic Encoder.
    '''
    def __init__(self, output_dim, input_length=None, composer_activation='linear',
                 return_mode='last_output', batch=None, weights=None, **kwargs):
        '''
        Arguments:
        output_dim (int)
        input_length (int)
        composer_activation (str): activation used in the MLP
        return_mode (str): One of last_output, all_outputs, output_and_memory
            This is analogous to the return_sequences flag in Keras' Recurrent.
            last_output returns only the last h_t
            all_outputs returns the whole sequence of h_ts
            output_and_memory returns the last output and the last memory concatenated
                (needed if this layer is followed by a MMA-NSE)
        weights (list): Initial weights --vj: what are these?
        '''
        self.output_dim = output_dim
        self.B = batch
        self.K = output_dim if output_dim != None else -1
        self.input_dim = output_dim  # Equation 2 in the paper makes this assumption.
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=3)]
        self.input_length = input_length
        self.L = input_length if input_length != None else -1
        self.composer_activation = composer_activation
        super(NSE, self).__init__(**kwargs)
        self.reader = LSTM(self.output_dim, dropout=0.0, recurrent_dropout=0.0, 
                           name="{}_reader".format(self.name))
        # TODO: Let the writer use parameter dropout and any consume_less mode.
        # Setting dropout to 0 here to eliminate the need for constants.
        # Setting consume_less to gpu to eliminate need for preprocessing
        self.writer = LSTM(self.output_dim, dropout=0.0, recurrent_dropout=0.0, 
                           name="{}_writer".format(self.name))
        self.composer = Dense(self.output_dim * 2, activation=self.composer_activation,
                              name="{}_composer".format(self.name))
        if return_mode not in ["last_output", "all_outputs", "output_and_memory"]:
            raise Exception("Unrecognized return mode: {}".format(return_mode))
        self.return_mode = return_mode


    def compute_output_shape(self, input_shape):
        return self.get_output_shape_for(input_shape)
    
    def get_output_shape_for(self, input_shape):
        input_length = input_shape[1]
        self.input_length = input_length
        if self.return_mode == "last_output": 
            return (input_shape[0], self.output_dim)
        elif self.return_mode == "all_outputs":
            return (input_shape[0], input_length, self.output_dim)
        else:
            # return_mode is output_and_memory. Output will be concatenated to memory.
            return (input_shape[0], input_length + 1, self.output_dim)

    def compute_mask(self, input, mask):
        if mask is None or self.return_mode == "last_output":
            return None
        elif self.return_mode == "all_outputs":
            return mask  # (batch_size, input_length)
        else:
            # Return mode is output_and_memory
            # Mask memory corresponding to all the inputs that are masked, and do not mask the output
            # (batch_size, input_length + 1)
            return K.cast(K.concatenate([K.zeros_like(mask[:, :1]), mask]), 'uint8')

    def get_composer_input_shape(self, input_shape):
        # Takes concatenation of output and memory summary
        return (input_shape[0], self.output_dim * 2)

    def get_writer_input_shape(self, input_shape):
        return (input_shape[0], 1, self.output_dim * 2)  # Will process one timestep at a time
    
    def get_reader_input_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        input_dim = input_shape[-1]

        reader_input_shape = self.get_reader_input_shape(input_shape)
        writer_input_shape = self.get_writer_input_shape(input_shape)
        composer_input_shape = self.get_composer_input_shape(input_shape)

        print("NSE writer input shape: {}".format(writer_input_shape), file=sys.stderr)        
        print("NSE reader input shape:{}".format(reader_input_shape) , file=sys.stderr)        
        print("NSE composer input shape: {}".format(composer_input_shape), file=sys.stderr)

        self.reader.build(reader_input_shape)
        self.writer.build(writer_input_shape)
        self.composer.build(composer_input_shape)

        # Aggregate weights of individual components for this layer.
        reader_weights = self.reader.trainable_weights
        writer_weights = self.writer.trainable_weights
        composer_weights = self.composer.trainable_weights

        #vj: Keras 2.2 -- 
        # (a) There are no trainable weights specific to this layer -- only those in reader, writer, composer
        #     (though ... what are initial_weights?)
        # (b) The compute graph built up in call should already know about r, w, c, and its weights.
        # So I believe nothing needs to be done here -- but not sure about this.
        # (Why did Keras 1.2 require trainable_weights to be set?)
 #       self.trainable_weights = reader_weights + writer_weights + composer_weights

        if self.initial_weights is not None:
            # self.set_weights(self.initial_weights)
            # vj Keras 2.2 conversion. Guessing the shape argument below.
            self.weights = self.add_weight(name='nse', shape=self.initial_weights.get_shape(), trainable=True)
            del self.initial_weights
        super(NSE, self).build(input_shape) 

    def get_initial_state(self, nse_input, input_mask=None):
        '''
        This method produces the 'read' mask for all timesteps
        and initializes the memory slot mem_0.

        Input: nse_input (batch_size, input_length, input_dim)
        Output: list[Tensors]:
                h_0 (batch_size, output_dim)
                c_0 (batch_size, output_dim)
                mem_0 (batch_size, output_dim)
                ...
                mem_l-1 (batch_size, output_dim)
 
        While this method simply copies input to mem_0, variants that inherit from this class can do
        something fancier.
        '''
        input_to_read = nse_input
        mem_0 = input_to_read


        flattened_mem_0 = flatten_memory(mem_0)
        initial_states = self.reader.get_initial_state(nse_input)
        initial_states += flattened_mem_0
        return initial_states

    def summarize_memory(self, o_t, mem_tm1):
        '''
        This method selects the relevant parts of the memory given the read output and summarizes the
        memory. Implements Equations 2-3 or 8-11 in the paper.
        '''
        # Selecting relevant memory slots, Equation 2
        z_t = K.softmax(K.sum(K.expand_dims(o_t, axis=1) * mem_tm1, axis=2)) 
        z_t = TF_PRINT(z_t, "z_t", expected_shape = [self.B, self.L])
        
        # Summarizing memory, Equation 3
        m_rt = K.sum(K.expand_dims(z_t, axis=2) * mem_tm1, axis=1)  
        m_rt = TF_PRINT(m_rt, "m_rt", expected_shape = [self.B, self.K])
        return z_t, m_rt

    def compose_memory_and_output(self, output_memory_list):
        '''
        This method takes a list of tensors and applies the composition function on their concatrnation.
        Implements equation 4 or 12 in the paper.
        '''
        # Composition, Equation 4
        c_t = self.composer.call(K.concatenate(output_memory_list)) 
        c_t = TF_PRINT(c_t, "c_t", expected_shape = [self.B, self.K*len(output_memory_list)])        
        return c_t

    def update_memory(self, z_t, h_t, mem_tm1):
        '''
        This method takes the attention vector (z_t), writer output (h_t) and previous timestep's memory 
        (mem_tm1) and updates the memory. Implements equations 6, 14 or 15.
        '''
        """ 
        The following is written assuming the equations in the paper are implemented as they are written:
        tiled_z_t_trans = K.tile(K.expand_dims(z_t,1), [1,self.output_dim,1])  # (batch_size, input_length, output_dim)
        input_length = K.shape(mem_tm1)[1]
        # (batch_size, input_length, output_dim)
#        tiled_h_t = K.permute_dimensions(K.tile(K.expand_dims(h_t, -1), [1,input_length]), (0, 2, 1))
        tiled_h_t = K.tile(K.expand_dims(h_t, -1), [1,1, input_length])
# Updating memory. First term in summation corresponds to selective forgetting and the second term to
        # selective addition. Equation 6.
        mem_t = mem_tm1 * (1 - tiled_z_t_trans) + tiled_h_t * tiled_z_t_trans  # (batch_size, input_length, output_dim)
        """

        """ 
        The following code assumes that mem_t is actually the transpose of what is in the paper.
        Implemented by simply wrapping a K.permute_dimensions(_, (0, 2, 1)) call around the original value.
        """
        tiled_z_t = K.permute_dimensions(K.tile(K.expand_dims(z_t,1), [1,self.output_dim,1]), (0, 2,1))  # (batch_size, input_length, output_dim)
        input_length = K.shape(mem_tm1)[1]
        # (batch_size, input_length, output_dim)
#        tiled_h_t = K.permute_dimensions(K.tile(K.expand_dims(h_t, -1), [1,input_length]), (0, 2, 1))
        tiled_h_t = K.permute_dimensions(K.tile(K.expand_dims(h_t, -1), [1,1, input_length]), (0, 2, 1))

        # Updating memory. First term in summation corresponds to selective forgetting and the second term to
        # selective addition. Equation 6.
        mem_t = mem_tm1 * (1 - tiled_z_t) + tiled_h_t * tiled_z_t 
        mem_t = TF_PRINT(mem_t, "mem_t", expected_shape=[self.B, self.L, self.K])
        
        return mem_t

    @staticmethod
    def split_states(states):
        # This method is a helper for the step function to split the states into reader states, memory and
        # awrite states.
        # increase by 1 because we add ht to the output states of step.
        # Three state tensors, then memory, then two state tensors at the end
        return states[:3], states[3:-2], states[-2:]

    @staticmethod
    def composes_states(read_states, memory, write_states):
        return read_states + memory + write_states
    
    @staticmethod
    def flatten_memory(mem):
        return tf.unstack(mem, axis=1)

    @staticmethod
    def unflatten_memory(flattened_mem):
        return tf.stack(flattened_mem, axis=1)

    def step(self, input_t, states):
        '''
        This method is a step function that updates the memory at each time step and produces
        a new output vector (Equations 1 to 6 in the paper).
        The memory_state is flattened because K.rnn requires all states to be of the same shape as the output,
        because it uses the same mask for the output and the states.

        vj: Note that all three elements of states returned below have the same *shape*, though
        flattened_mem_tm1 has different value for the second dimension, hence is not the same size.
        Keras 1.2.1 K.rnn requires all the state tensors to be of the same shape *and* size.
     
        Inputs:
            input_t (batch_size, input_dim)
            states (list[Tensor])
                flattened_mem_tm1 (batch_size, input_length * output_dim)
                writer_h_tm1 (batch_size, output_dim)
                writer_c_tm1 (batch_size, output_dim)

        Outputs:
            h_t (batch_size, output_dim)
            flattened_mem_t (batch_size, input_length * output_dim)
        '''
        input_t = TF_PRINT(input_t, "input_t", expected_shape=[self.B, self.K])
        
        reader_states, flattened_mem_tm1, writer_states = split_states(states)

        # Reshape the memory
        mem_tm1 = unflatten_memory(flattened_mem_tm1) 
        mem_tm1 = TF_PRINT(mem_tm1, "mem_tm1", expected_shape=[self.B, self.L, self.K])

        # vj Keras 2.2
        # Do not have get_constants any more.
        # self.reader.get_constants(input_t)  # Does not depend on input_t, see init.
        reader_constants = [] 
        reader_states = reader_states[:2] + (reader_constants) + reader_states[2:]
        # o_t, reader_c_t: (batch_size, output_dim)
        o_t, [_, reader_c_t] = self.reader.step(input_t, reader_states)  

        o_t = TF_PRINT(o_t, "o_t", expected_shape=[self.B, self.K])
        reader_c_t = TF_PRINT(reader_c_t, "reader_c_t", expected_shape=[self.B, self.K])
        
        z_t, m_rt = self.summarize_memory(o_t, mem_tm1)
        c_t = self.compose_memory_and_output([o_t, m_rt])


        # Collecting the necessary variables to directly call writer's step function.
        writer_constants = self.writer.get_constants(c_t)  # returns dropouts for W and U (all 1s, see init)
        writer_states += tuple(writer_constants)

        # Making a call to writer's step function, Equation 5
        # h_t, writer_c_t: (batch_size, output_dim)
        h_t, [_, writer_c_t] = self.writer.step(c_t, writer_states)  

        h_t = TF_PRINT(h_t, "h_t", expected_shape=[self.B, self.K])
        writer_c_t = TF_PRINT(writer_c_t, "writer_c_t", expected_shape=[self.B, self.K])

        mem_t = self.update_memory(z_t, h_t, mem_tm1)

        # vj TODO: The first state returned at time t should be the value of the output at time t-1.
        # so that shouldbe h_(t-1). Where do we get this from?
        # Need to fix the initial state to have the same shape as well.
        # For now, pass h_t.

        return h_t, compose_states([h_t, o_t, reader_c_t], flatten_memory(mem_t), [ h_t, writer_c_t])


    def loop(self, x, initial_states, mask):
        # This is a separate method because Ontoaware variants will have to override this to make a call
        # to changingdim rnn.

        last_output, all_outputs, last_states = K.rnn(self.step, x, initial_states, mask=mask)
        last_output = TF_PRINT(last_output, "loop.last_output")
        all_outputs = TF_PRINT(all_outputs, "loop.all_outputs")
        for i, state in enumerate(last_states):
            state = TF_PRINT(state, "loop.laststates.{}".format(i))
            #        last_states = TF_PRINT(last_states, "loop.last_states" )                
        return last_output, all_outputs, last_states

    def call(self, x, mask=None):
        # input_shape = (batch_size, input_length, input_dim). This needs to be defined in build.
        mask = TF_PRINT(mask, "mask")        
        initial_read_states = self.get_initial_state(x, mask)

        #vj duplicate the first guy -- TODO: check if a stateful LSTM should be used.
        initial_read_states = [initial_read_states[0]] + initial_read_states

        fake_writer_input = K.expand_dims(initial_read_states[0], axis=1)  # (batch_size, 1, output_dim)
        fake_writer_input = TF_PRINT(fake_writer_input, "fake_writer_input", expected_shape=[self.B, 1, self.K])
        
        initial_write_states = self.writer.get_initial_state(fake_writer_input)  # h_0 and c_0 of the writer LSTM
        initial_states = initial_read_states + initial_write_states

        # last_output: (batch_size, output_dim)
        # all_outputs: (batch_size, input_length, output_dim)
        # last_states:
        #       last_memory_state: (batch_size, input_length, output_dim)
        #       last_output
        #       last_writer_ct
        last_output, all_outputs, last_states = self.loop(x, initial_states, mask)
        last_memory = last_states[0]
        
        if self.return_mode == "last_output":
            return last_output
        elif self.return_mode == "all_outputs":
            return all_outputs
        else:
            # return mode is output_and_memory
            expanded_last_output = K.expand_dims(last_output, axis=1)  # (batch_size, 1, output_dim)
            expanded_last_output = TF_PRINT(expanded_last_output, "expanded_last_output",
                                            expected_size=[self.B, 1, self.K])
            # (batch_size, 1+input_length, output_dim)
            result = K.concatenate([expanded_last_output, last_memory], axis=1)
            result = TF_PRINT(result, "result", expected_size=[self.B, 1+self.L, self.K])
            return result

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'input_length': self.input_length,
                  'composer_activation': self.composer_activation,
                  'return_mode': self.return_mode}
        base_config = super(NSE, self).get_config()
        config.update(base_config)
        return config


class InputMemoryMerger(Layer):
    '''
    This layer taks as input, the memory part of the output of a NSE layer, and the embedded input to a MMANSE
    layer, and prepares a single input tensor for MMANSE that is a concatenation of the first sentence's memory
    and the second sentence's embedding.
    This is a concrete layer instead of a lambda function because we want to support masking.
    TODO: vj Check if Keras 2.2 supports masking in Lambda layers.
    '''
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(InputMemoryMerger, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[1][0], input_shapes[1][1]*2, input_shapes[1][2])

    def compute_output_shape(self, input_shape):
        return self.get_output_shape_for(input_shape)
    
    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        if mask is None:
            return None
        elif mask == [None, None]:
            return None
        else:
            memory_mask, mmanse_embed_mask = mask
            return K.concatenate([mmanse_embed_mask, memory_mask], axis=1)  # (batch_size, nse_input_length * 2)
        
    def call(self, inputs, mask=None):
        shared_memory = inputs[0]
        mmanse_embed_input = inputs[1]  # (batch_size, nse_input_length, output_dim)
        return K.concatenate([mmanse_embed_input, shared_memory], axis=1)

class OutputSplitter(Layer):
    '''
    This layer takes the concatenation of output and memory from NSE and returns either the output or the
    memory.
    TODO: CHeck if this should be a Lambda layer.
    '''
    def __init__(self, return_mode, **kwargs):
        self.supperots_masking = True
        if return_mode not in ["output", "memory"]:
            raise Exception("Invalid return mode: %s" % return_mode)
        self.return_mode = return_mode
        super(OutputSplitter, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        if self.return_mode == "output":
            return (input_shape[0], input_shape[2])
        else:
            # Return mode is memory.
            # input contains output and memory concatenated along the second dimension.
            return (input_shape[0], input_shape[1] - 1, input_shape[2])

    def compute_output_shape(self, input_shape):
        return self.get_output_shape_for(input_shape)
    
    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        if self.return_mode == "output" or mask is None:
            return None
        else:
            # Return mode is memory and mask is not None
            return mask[:, 1:]  # (batch_size, nse_input_length)

    def call(self, inputs, mask=None):
        if self.return_mode == "output":
            return inputs[:, 0, :]  # (batch_size, output_dim)
        else:
            return inputs[:, 1:, :]  # (batch_size, nse_input_length, output_dim)

    def get_config(self):
        config = {"return_mode": self.return_mode}
        base_config = super(OutputSplitter, self).get_config()
        config.update(base_config)
        return config

class MultipleMemoryAccessNSE(NSE):
    '''
    MultipleMemoryAccessNSE is very similar to the simple NSE. The difference is that along with the sentence
    memory, it has access to one (or multiple) additional memory. The operations on the additional memory are
    exactly the same as the original memory. The additional memory is initialized from the final timestep of
    a different NSE, and the composer will take as input the concatenation of the reader output and summaries
    of both the memories.
    '''
    #TODO: This is currently assuming we need access to one additional memory. Change it to an arbitrary number.
    @overrides
    def get_output_shape_for(self, input_shape):
        # This class has twice the input length as an NSE due to the concatenated input. Pass the right size
        # to NSE's method to get the right putput shape.
        nse_input_shape = (input_shape[0], input_shape[1]/2, input_shape[2])
        return super(MultipleMemoryAccessNSE, self).get_output_shape_for(nse_input_shape)

    def get_reader_input_shape(self, input_shape):
        return (input_shape[0], input_shape[1]/2, self.output_dim)

    def get_composer_input_shape(self, input_shape):
        return (input_shape[0], self.output_dim * 3)

    @overrides
    def get_initial_state(self, nse_input, input_mask=None):
        '''
        Read input in MMA-NSE will be of shape (batch_size, read_input_length*2, input_dim), a concatenation 
        of the actual input to this NSE and the output from a different NSE. The latter will be used to 
        initialize the shared memory. The former will be passed to the read LSTM and also used to initialize 
        the current memory.
        '''
        input_length = K.shape(nse_input)[1]
        read_input_length = input_length/2
        input_to_read = nse_input[:, :read_input_length, :]
        initial_shared_memory = K.batch_flatten(nse_input[:, read_input_length:, :])
        mem_0 = K.batch_flatten(input_to_read)
        o_mask = self.reader.compute_mask(input_to_read, input_mask)
        reader_states = self.reader.get_initial_state(nse_input)
        initial_states = reader_states + [mem_0, initial_shared_memory]
        return initial_states, o_mask

    # vj: TODO check this logic continues to work for Keras 2.2
    @overrides
    def step(self, input_t, states):
        reader_states = states[:2]
        flattened_mem_tm1, flattened_shared_mem_tm1 = states[2:4]
        writer_h_tm1, writer_c_tm1 = states[4:]
        input_mem_shape = K.shape(flattened_mem_tm1)
        mem_shape = (input_mem_shape[0], input_mem_shape[1]/self.output_dim, self.output_dim)
        mem_tm1 = K.reshape(flattened_mem_tm1, mem_shape)
        shared_mem_tm1 = K.reshape(flattened_shared_mem_tm1, mem_shape)
        reader_constants = () # self.reader.get_constants(input_t)
        reader_states += tuple(reader_constants)
        o_t, [_, reader_c_t] = self.reader.step(input_t, reader_states)
        z_t, m_rt = self.summarize_memory(o_t, mem_tm1)
        shared_z_t, shared_m_rt = self.summarize_memory(o_t, shared_mem_tm1)
        c_t = self.compose_memory_and_output([o_t, m_rt, shared_m_rt])

        # Collecting the necessary variables to directly call writer's step function.
        writer_constants = self.writer.get_constants(c_t)  # returns dropouts for W and U (all 1s, see init)
        writer_states = [writer_h_tm1, writer_c_tm1] + writer_constants

        # Making a call to writer's step function, Equation 5
        h_t, [_, writer_c_t] = self.writer.step(c_t, writer_states)  # h_t, writer_c_t: (batch_size, output_dim)
        mem_t = self.update_memory(z_t, h_t, mem_tm1)
        shared_mem_t = self.update_memory(shared_z_t, h_t, shared_mem_tm1)
        return h_t, [o_t, reader_c_t, K.batch_flatten(mem_t), K.batch_flatten(shared_mem_t), h_t, writer_c_t]

