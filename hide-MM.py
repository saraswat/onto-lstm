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
    def get_initial_states(self, nse_input, input_mask=None):
        '''
        Read input in MMA-NSE will be of shape (batch_size, read_input_length*2, input_dim), a concatenation of
        the actual input to this NSE and the output from a different NSE. The latter will be used to initialize
        the shared memory. The former will be passed to the read LSTM and also used to initialize the current
        memory.
        '''
        input_length = K.shape(nse_input)[1]
        read_input_length = input_length/2
        print("vj: get_initial_states nse_input:{}".format(nse_input))
        input_to_read = nse_input[:, :read_input_length, :]
        initial_shared_memory = K.batch_flatten(nse_input[:, read_input_length:, :])
        mem_0 = K.batch_flatten(input_to_read)
        o_mask = self.reader.compute_mask(input_to_read, input_mask)
        reader_states = self.reader.get_initial_states(nse_input)
        initial_states = reader_states + [mem_0, initial_shared_memory]
        return initial_states, o_mask

    @overrides
    def step(self, input_t, states):
        reader_states = states[:2]
        flattened_mem_tm1, flattened_shared_mem_tm1 = states[2:4]
        writer_h_tm1, writer_c_tm1 = states[4:]
        input_mem_shape = K.shape(flattened_mem_tm1)
        mem_shape = (input_mem_shape[0], input_mem_shape[1]/self.output_dim, self.output_dim)
        mem_tm1 = K.reshape(flattened_mem_tm1, mem_shape)
        shared_mem_tm1 = K.reshape(flattened_shared_mem_tm1, mem_shape)
        reader_constants = self.reader.get_constants(input_t)
        reader_states += reader_constants
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

