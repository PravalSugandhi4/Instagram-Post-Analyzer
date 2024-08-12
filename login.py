import instaloader
import re

my_list = []

# Initialize Instaloader
L = instaloader.Instaloader()

username=input("Enter Your Instagram User Name:- ")
password=input("Enter Your Instagram Password:- ")
# Login (optional, but recommended for private profiles)
try:
    L.login(username, password)
    print("\nLogin Sucessful")
except Exception as e:
    print(f"\nFailed to login: {e}")
    exit(1)

# Function to extract shortcode from URL
def extract_shortcode(url):
    match = re.search(r'instagram\.com/p/([^/]+)/', url)
    if match:
        print("\nShort code generated sucessfully")
        return match.group(1)
    else:
        raise ValueError("\nInvalid URL format")
        exit(1)

# Replace this with your actual Instagram post URL
post_url = input("Enter Post URL:-")
try:
    shortcode = extract_shortcode(post_url)
    print(f"\nExtracted shortcode: {shortcode}")

    # Load the post using the shortcode
    post = instaloader.Post.from_shortcode(L.context, shortcode)
    
    # Print the comments
    print(f'Post: {post.shortcode}')
    comments = list(post.get_comments())
    if comments:
        for comment in comments:
           my_list.append(comment.text)
    else:
        print("No comments found on this post.")
        exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
#----------------------------------------------------------------------------------------------------
L.close()
print("Logged out successfully.")
print(my_list)
