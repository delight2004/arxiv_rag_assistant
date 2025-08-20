import arxiv
from pathlib import Path
from grobid_client.grobid_client import GrobidClient
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---- Setup and Download ----
client_gro = GrobidClient(grobid_server="http://localhost:8070")
client = arxiv.Client()
search_query = arxiv.Search(query="Attention Is All You Need", id_list=["1706.03762"], max_results=1)
search_results = client.results(search_query)

downloads_dir = Path("data/downloaded_papers")
downloads_dir.mkdir(exist_ok=True)


    # --- Consolidated Processing Loop
for article in search_results:
    pdf_path_str = article.download_pdf(dirpath=downloads_dir)
    pdf_path = Path(pdf_path_str)
    print(f"Downloaded: '{article.title}'")
    print(f"PDF saved to: {pdf_path}")
    print("-" * 50)

    # Process the PDF with GROBID
    try:

        client_gro.process(
            "processFulltextDocument",
            str(downloads_dir),  # must be a string path
            n=1,
            generateIDs=False,
            consolidate_header=True,
            consolidate_citations=False,
            include_raw_citations=False,
            include_raw_affiliations=False,
            tei_coordinates=False,
            segment_sentences=False,
            force=True
        )

        # construct and verify xml path
        xml_filename = f"{pdf_path.stem}.grobid.tei.xml"
        xml_path = downloads_dir / xml_filename

        if xml_path.exists():
            print(f"XML file created: {xml_path}")

            # --- XML Parsing ---
            with open(xml_path, "r", encoding="utf-8") as file:
                content = file.read()

                #parse xml with BeautifulSoup
                soup = BeautifulSoup(content, "xml")

                #extract <title>
                title = soup.find("title").get_text() if soup.find("title") else "Title not found"

                #extract abstract
                abstract = soup.find("abstract").get_text() if soup.find("abstract") else "Abstract not found"

                #extract all paragraphs
                paragraphs = [p.get_text() for p in soup.find_all("p")]

                print("\n---- Extracted Metadata ----")
                print(f"Title: {title}")
                print(f"Abstract: {abstract[:300]}...")
                print(f"Number of paragraphs: {len(paragraphs)}")
                print(f"First paragraph: {paragraphs[0][:150]}...")
                print("-" * 50)

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )

                #now split the paragraphs into chunks
                docs = text_splitter.create_documents(paragraphs)

                print("\n--- Chunking Results ---")
                print(f"Number of chunks: {len(docs)}")
                print(f"First chunk: {docs[0]}")
                print("-" * 50)

                # Instantiate the embedding model/instantiate huggingface embeddings to load the model
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
                embeddings = HuggingFaceEmbeddings(model_name=model_name)

                # create the FAISS vector database from the chunks
                db = FAISS.from_documents(docs, embeddings)

                #save the database to disk
                db_path = downloads_dir / "faiss_index"
                db.save_local(db_path)

                print("\n--- Vector Database ---")
                print(f"FAISS index created and saved to: {db_path}")
                print("-" * 50)

        else:
            print("XML file not found.")           
        
    except Exception as e:
        print(f"An error occurred during GROBID processing or XML parsing: {e}")