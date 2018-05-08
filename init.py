import pickle

from user import User, Post

if __name__ == '__main__':
    user_data = "../umd_reddit_suicidewatch_dataset/user_info"

    pickle_obj = open(user_data, 'rb')
    user_dict = pickle.load(pickle_obj)

    n_posts, sw_posts = 0, 0
    for n_users, (_,user) in enumerate(user_dict.items()):
        for post in user.posts:
            n_posts += 1
            if post.subreddit == 'SuicideWatch':
                sw_posts += 1

    print(n_posts, sw_posts)
