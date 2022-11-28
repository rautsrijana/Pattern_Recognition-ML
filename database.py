# DataBase
import sqlite3

conn = sqlite3.connect('usersdata.db', check_same_thread=False)
c = conn.cursor()

# Functions

def create_usertable():
	"""
	function to create a table
	"""
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
	"""
	function that add the different users.
	"""
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	"""
	login Credentials
	"""
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data



def view_all_users():
	"""
	View all the users
	"""
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data