from retrieval import load_chunks

chunks = load_chunks()
print("Nombre total de chunks :", len(chunks))
print("\nExemples d'identifiants :")
for chunk in chunks[:10]:
    print(repr(chunk.get("article_id")), repr(chunk.get("chunk_id")))

print("\nChunks correspondant à l'article 5 :")
article5 = [
    chunk for chunk in chunks
    if "5" in str(chunk.get("article_id", ""))
    and "Article_5" in str(chunk.get("chunk_id", ""))
]
print("Nombre trouvé :", len(article5))
for chunk in article5[:5]:
    print(chunk["article_id"], chunk["chunk_id"])
    print(chunk["text"][:150])
    print("---")
