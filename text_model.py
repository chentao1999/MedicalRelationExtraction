# coding=utf-8
'''
This script is based on https://github.com/cjymz886/text_bert_cnn and https://github.com/google-research/bert.
'''
import  tensorflow as tf
import collections
from  bert import modeling
from bert import tokenization

class InputExample(object):
    """A single training/test example for simple sequence classification."""
    
    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. 
          text_b: string. The untokenized text of the second sequence.
          label: string. The label of the example. This should be specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def convert_examples_to_features(examples,label_list, max_seq_length,tokenizer):
    """Loads a data file into a list of `InputBatch`s.
    Convert examples into token form as input of BERT model.
    """
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    input_data=[]
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 3:
            tf.logging.info("*** Example ***")
            tf.logging.info("guid: %s" % (example.guid))
            tf.logging.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

        features = collections.OrderedDict()
        features["input_ids"] = input_ids
        features["input_mask"] = input_mask
        features["segment_ids"] = segment_ids
        features["label_ids"] =label_id
        input_data.append(features)

    return input_data

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def batch_iter(input_data,batch_size):
    """Batch feed the four tokens form variables of the sample into the model.s
    """
    batch_ids,batch_mask,batch_segment,batch_label=[],[],[],[]
    for features in input_data:
        if len(batch_ids) == batch_size:
            yield batch_ids,batch_mask,batch_segment,batch_label
            batch_ids, batch_mask, batch_segment, batch_label = [], [], [], []

        batch_ids.append(features['input_ids'])
        batch_mask.append(features['input_mask'])
        batch_segment.append(features['segment_ids'])
        batch_label.append(features['label_ids'])

    if len(batch_ids) != 0:
        yield batch_ids, batch_mask, batch_segment, batch_label

def optimistic_restore(session, save_file):
    """Load bert model. """
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for
                      var in tf.global_variables()
                      if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x: x.name.split(':')[0],tf.global_variables()),tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                # print("going to restore.var_name:",var_name,";saved_var_name:",saved_var_name)
                restore_vars.append(curr_var)
            else:
                print("variable not trained.var_name:",var_name)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)
    
class TextCNN(object):

    def __init__(self,config):
        '''Get the hyperparameters and the five variables needed by the model, i.e. input_ids，input_mask，segment_ids，labels，keep_prob'''
        self.config=config
        self.bert_config = modeling.BertConfig.from_json_file(self.config.bert_config_file)

        self.input_ids=tf.placeholder(tf.int64,shape=[None,self.config.seq_length],name='input_ids')
        self.input_mask=tf.placeholder(tf.int64,shape=[None,self.config.seq_length],name='input_mask')
        self.segment_ids=tf.placeholder(tf.int64,shape=[None,self.config.seq_length],name='segment_ids')
        self.labels=tf.placeholder(tf.int64,shape=[None,],name='labels')
        self.keep_prob=tf.placeholder(tf.float32,name='dropout')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.cnn()

    def cnn(self):
        '''Get the final token-level output of BERT model using get_sequence_output function, and use it as the input embeddings of CNN model.
        '''
        with tf.name_scope('bert'):
            bert_model = modeling.BertModel(
                config=self.bert_config,
                is_training=self.config.is_training,
                input_ids=self.input_ids,
                input_mask=self.input_mask,
                token_type_ids=self.segment_ids,
                use_one_hot_embeddings=self.config.use_one_hot_embeddings)
            embedding_inputs= bert_model.get_sequence_output()

        '''Use three convolution kernels to do convolution and pooling, and concat the three resutls.'''
        with tf.name_scope('conv'):
            pooled_outputs = []
            for i, filter_size in enumerate(self.config.filter_sizes):
                with tf.variable_scope("conv-maxpool-%s" % filter_size,reuse=False):
                    conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, filter_size,name='conv1d')
                    pooled = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
                    pooled_outputs.append(pooled)

            num_filters_total = self.config.num_filters * len(self.config.filter_sizes)
            h_pool = tf.concat(pooled_outputs, 1)
            outputs = tf.reshape(h_pool, [-1, num_filters_total])

        '''Add full connection layer and dropout layer'''
        with tf.name_scope('fc'):
            fc=tf.layers.dense(outputs,self.config.hidden_dim,name='fc1')
            fc = tf.nn.dropout(fc, self.keep_prob)
            fc=tf.nn.relu(fc)

        '''logits'''
        with tf.name_scope('logits'):
            self.logits = tf.layers.dense(fc, self.config.num_labels, name='logits')
            self.prob = tf.nn.softmax(self.logits)
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)

        '''Calculate loss. Convert predicted labels into one hot form. '''
        with tf.name_scope('loss'):
            log_probs = tf.nn.log_softmax(self.logits, axis=-1)
            one_hot_labels = tf.one_hot(self.labels, depth=self.config.num_labels, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            self.loss = tf.reduce_mean(per_example_loss)

        '''optimizer'''
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.config.lr)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.clip)
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        '''accuracy'''
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(self.labels, self.y_pred_cls)
            self.acc=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
