from __future__ import division
from __future__ import print_function

import os.path
import re
import csv
import tensorflow as tf
from chi2_distance import *

FLAGS = tf.app.flags.FLAGS

# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.
tf.app.flags.DEFINE_string(
    'model_dir', '/tmp/imagenet',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_string('image_file', '',   ### here you can indicate the image file !!
                           """Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long

class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.
    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.
    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


class DeepLearningSearcher(object):
	def __init__(self, deep_learning_data):
		# store our index path
		self.deep_learning_data = deep_learning_data
		self.dp_list = []
		self.node_lookup = NodeLookup()

		with open(self.deep_learning_data) as f:
			# initialize the CSV reader
			reader = csv.reader(f)

			# loop over the rows in the index
			for row in reader:
				self.dp_list.append(row)
				
			# close the reader
			f.close()

		"""Creates a graph from saved GraphDef file and returns a saver."""
		# Creates graph from saved graph_def.pb.
		with tf.gfile.FastGFile(os.path.join(
			FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			_ = tf.import_graph_def(graph_def, name='')

		with tf.Session() as sess:
		# Some useful tensors:
		# 'softmax:0': A tensor containing the normalized prediction across
		#   1000 labels.
		# 'pool_3:0': A tensor containing the next-to-last layer containing 2048
		#   float description of the image.
		# 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
		#   encoding of the image.
		# Runs the softmax tensor by feeding the image_data as input to the graph.
			self.softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
			self.sess = sess


	def run_inference_on_image(self, input_path):
	    # extract the image ID (i.e. the unique filename) from the image
	    # path and load the image itself

	    # describe the image
	    image_data = tf.gfile.FastGFile(input_path, 'rb').read()
	    predictions = self.sess.run(self.softmax_tensor,
	                           {'DecodeJpeg/contents:0': image_data})
	    predictions = np.squeeze(predictions)

	    dp_match_results = self.search_by_deep_learning(predictions)


	    visual_concepts = []

	    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]

	    for node_id in top_k:
			human_string = self.node_lookup.id_to_string(node_id)
			score = predictions[node_id]
			visual_concepts.append([human_string, score])

	    return (dp_match_results, visual_concepts)

	def search_by_deep_learning(self, queryFeatures, weight=1):
		results = {}
		for row in self.dp_list:
			features = np.array(row[1:], dtype = np.dtype(float))
			d = np.linalg.norm(queryFeatures - features)

			results[row[0]] = d * weight
		return results


