import tweepy
import datetime

# super secret information (please don't steal)
API_KEY = 'DnzKfdItAT7OENyBmQCTU3s9t'
API_SECRET_KEY = 'QWeDZvBzXU1bPyZmeOvV2VB0PYqN68c9CQ2RZMlytjXgYIq4uw'
ACCESS_TOKEN = '1571766349915095040-kKcNiZuq6DsvAuusGovTawFs2YAODo'
ACCESS_TOKEN_SECRET = 'Rbx8oSJLX1K2LPMrMnuIWTDbUzuOh7BSYVQcWuJ7j5tUX'
BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAKKtwgEAAAAABIx5ASCHFouTkyf21ITPNBE9%2BtU%3Ds9UlN30WMD59eJsjARuuAymqR4Ydjsf18uuuaBmaI3r6CjluKj'

# Set up Tweepy authentication
auth = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

# Create an API object to interact with Twitter
api = tweepy.API(auth)

# Verify credentials
try:
    api.verify_credentials()
    print("Authentication successful")
except:
    print("Authentication failed")



def get_twitter_mentions(queryIn, start_date, end_date):
    # authenticate with credentials
    client = tweepy.Client(
        bearer_token=BEARER_TOKEN,
        consumer_key=API_KEY,
        consumer_secret=API_SECRET_KEY,
        access_token=ACCESS_TOKEN,
        access_token_secret=ACCESS_TOKEN_SECRET
    )

    # Initialize list of daily mention counts
    daily_mentions = []

    # convert start and end dates to datetime objects
    current_date = datetime.datetime.strptime(start_date, "%d-%m-%Y")
    end_date = datetime.datetime.strptime(end_date, "%d-%m-%Y")

    # loop through each day in the range
    while current_date <= end_date:
        # search tweets mentioning the company for this day using Twitter API
        tweets = client.search_all_tweets(
            query=queryIn,
            start_time=current_date.isoformat() + "Z",
            end_time=next_day.isoformat() + "Z",
            max_results=100
        )

        # count the number of tweets for the day
        tweet_count = len(tweets.data) if tweets.data else 0
        # append to data object
        daily_mentions.append(tweet_count)

        # move to the next day
        next_day = current_date + datetime.timedelta(days=1)
        current_date = next_day

    return daily_mentions

get_twitter_mentions("Commonwealth Bank", "01-01-2020", "01-01-2022")
