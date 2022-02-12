import offline_navigation, online_navigation

if __name__ == "__main__":
	offline_or_online = input("Offline or Online Navigation? (1/2) ")
	while offline_or_online != "1" and offline_or_online != "2":
		offline_or_online = input("Invalid choice. Choose either offline (1) or online (2) ")
	if offline_or_online == "1":
		offline_navigation.run_navigation()
	else:
		online_navigation.run_navigation(random_or_guided="guided")