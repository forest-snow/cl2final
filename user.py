class Post:
	def __init__(self, postid, text, timestamp):
		self.postid = postid
		self.text = text
		self.timestamp = timestamp

class User:
	"""
	User encapsulates information about a user and his/her posts.
	:param userid: int id
	:param label: 1 for true positive (sw), -1 for control, 0 for sw_cd, -2 for unknown
	:param posts: list of posts
	"""
	def __init__(self, userid, label):
		self.userid = userid
		self.label = label
		self.posts = []

	def add_post(self, post):
		self.posts.append(post)
