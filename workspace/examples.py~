import freesound
c = freesound.FreesoundClient()
c.set_token("<YOUR_API_KEY_HERE>","token")

# Get sound info example
print "Sound info:"
print "-----------"
s = c.get_sound(96541)
print "Getting sound: " + s.name
print "Url: " + s.url
print "Description: " + s.description
print "Tags: " + str(s.tags)
print "\n"

# Get sound analysis example
print "Get analysis:"
print "-------------"
analysis = s.get_analysis()
mfcc = analysis.lowlevel.mfcc.mean
print "Mfccs: " + str(mfcc)
print "\n"

# Get similar sounds example
print "Similar sounds: "
print "---------------"
results_pager = s.get_similar()
for i in range(0, len(results_pager.results)):
    sound = results_pager[i]
    print "\t- " + sound.name + " by " + sound.username
print "\n"

# Search Example
print "Searching for 'violoncello':"
print "----------------------------"
results_pager = c.text_search(query="violoncello",filter="tag:tenuto duration:[1.0 TO 15.0]",sort="rating_desc",fields="id,name,previews,username")
print "Num results: " + str(results_pager.count)
print "\t ----- PAGE 1 -----"
for i in range(0, len(results_pager.results)):
    sound = results_pager[i]
    print "\t- " + sound.name + " by " + sound.username
print "\t ----- PAGE 2 -----"
results_pager = results_pager.next_page()
for i in range(0, len(results_pager.results)):
    sound = results_pager[i]
    print "\t- " + sound.name + " by " + sound.username
print "\n"
results_pager[0].retrieve_preview('.')

# Content based search example
print "Content based search:"
print "---------------------"
results_pager = c.content_based_search(descriptors_filter="lowlevel.pitch.var:[* TO 20]",
    target='lowlevel.pitch_salience.mean:1.0 lowlevel.pitch.mean:440')

print "Num results: " + str(results_pager.count)
for i in range(0, len(results_pager.results)):
    sound = results_pager[i]
    print "\t- " + sound.name + " by " + sound.username
print "\n"


# Getting sounds from a user example
print "User sounds:"
print "-----------"
u = c.get_user("Jovica")
print "User name: " + u.username
results_pager = u.get_sounds()
print "Num results: " + str(results_pager.count)
print "\t ----- PAGE 1 -----"
for i in range(0, len(results_pager.results)):
    sound = results_pager[i]
    print "\t- " + sound.name + " by " + sound.username
print "\t ----- PAGE 2 -----"
results_pager = results_pager.next_page()
for i in range(0, len(results_pager.results)):
    sound = results_pager[i]
    print "\t- " + sound.name + " by " + sound.username
