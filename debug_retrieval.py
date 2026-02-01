from index.search import search

hits = search("driver obligations", k=10, where={"season": 2021})
print("hits:", len(hits))
for h in hits[:3]:
    print(h["meta"]["source"], h["meta"]["season"], h["meta"]["page"])
