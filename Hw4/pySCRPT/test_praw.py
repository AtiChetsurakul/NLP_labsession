import praw
from secKey import ret_id


if __name__ == '__main__':

    client, secret, _ = ret_id()
    reddit = praw.Reddit(client_id=client,

                         client_secret=secret,

                         user_agent='ati')

    print(reddit.read_only)  # Output: True

    subreddit = reddit.subreddit('apple')
    topics = [*subreddit.top(limit=50)]  # top posts all time
    # print(len(topics))
    fifty_sen = title = [n.title for n in topics]
