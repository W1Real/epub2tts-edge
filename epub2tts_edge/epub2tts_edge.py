import argparse
import asyncio
import concurrent.futures
import os
import re
import subprocess
import time
import warnings
import sys
from tqdm import tqdm

from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub
import edge_tts
from lxml import etree
import nltk
from nltk.tokenize import sent_tokenize
from PIL import Image
from pydub import AudioSegment
import zipfile

namespaces = {
   "calibre":"http://calibre.kovidgoyal.net/2009/metadata",
   "dc":"http://purl.org/dc/elements/1.1/",
   "dcterms":"http://purl.org/dc/terms/",
   "opf":"http://www.idpf.org/2007/opf",
   "u":"urn:oasis:names:tc:opendocument:xmlns:container",
   "xsi":"http://www.w3.org/2001/XMLSchema-instance",
}

warnings.filterwarnings("ignore", module="ebooklib.epub")

def ensure_punkt():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")

def chap2text_epub(chap, encoding="utf-8"):
    blacklist = ["[document]", "noscript", "header", "html", "meta", "head", "input", "script"]
    paragraphs = []
    
    # ---------------------------------------------------------
    # THE ENCODING FIX:
    # Intercept raw bytes and force decode them BEFORE BeautifulSoup 
    # can ruin them with bad guesses.
    # ---------------------------------------------------------
    if isinstance(chap, bytes):
        try:
            chap = chap.decode(encoding)
        except UnicodeDecodeError:
            print(f"\n⚠️ WARNING: Failed to decode chapter using '{encoding}'. Falling back to utf-8 with error replacement.")
            chap = chap.decode('utf-8', errors='replace')
            
    soup = BeautifulSoup(chap, "html.parser")
    
    # More robust HTML title fallback
    chapter_title = soup.find("h1")
    if not chapter_title: chapter_title = soup.find("h2")
    if not chapter_title: chapter_title = soup.find("title")
        
    if chapter_title and chapter_title.text.strip():
        chapter_title_text = chapter_title.text.strip()
    else:
        chapter_title_text = None
        
    for a in soup.findAll("a", href=True):
        if not any(char.isalpha() for char in a.text):
            a.extract()
            
    chapter_paragraphs = soup.find_all("p")
    if len(chapter_paragraphs) == 0:
        chapter_paragraphs = soup.find_all("div")
        
    for p in chapter_paragraphs:
        paragraph_text = "".join(p.strings).strip()
        paragraphs.append(paragraph_text)
        
    return chapter_title_text, paragraphs

def get_epub_cover(epub_path):
    try:
        with zipfile.ZipFile(epub_path) as z:
            t = etree.fromstring(z.read("META-INF/container.xml"))
            rootfile_path =  t.xpath("/u:container/u:rootfiles/u:rootfile", namespaces=namespaces)[0].get("full-path")
            t = etree.fromstring(z.read(rootfile_path))
            cover_meta = t.xpath("//opf:metadata/opf:meta[@name='cover']", namespaces=namespaces)
            if not cover_meta:
                return None
            cover_id = cover_meta[0].get("content")
            cover_item = t.xpath("//opf:manifest/opf:item[@id='" + cover_id + "']", namespaces=namespaces)
            if not cover_item:
                return None
            cover_href = cover_item[0].get("href")
            cover_path = os.path.join(os.path.dirname(rootfile_path), cover_href)
            if os.name == 'nt' and '\\' in cover_path:
                cover_path = cover_path.replace("\\", "/")
            return z.open(cover_path)
    except FileNotFoundError:
        print(f"Could not get cover image of {epub_path}")

def export(book, sourcefile, encoding="utf-8"):
    book_contents = []
    cover_image = get_epub_cover(sourcefile)
    
    image_filename = sourcefile.replace(".epub", ".png")
    if cover_image is not None:
        image = Image.open(cover_image)
        image.save(image_filename)
        print(f"Cover image saved to {image_filename}")
    else:
        print("No cover found in epub.")

    # Parse TOC.NCX for real chapter titles
    toc_map = {}
    for item in book.get_items():
        if type(item) == ebooklib.epub.EpubNcx:
            try:
                ncx_soup = BeautifulSoup(item.get_content(), "html.parser")
                for nav in ncx_soup.find_all("navpoint"):
                    text_node = nav.find("text")
                    content_node = nav.find("content")
                    if text_node and content_node and content_node.get("src"):
                        src = content_node.get("src").split("#")[0]
                        if src not in toc_map:
                            toc_map[src] = text_node.text.strip()
            except Exception as e:
                print(f"Warning: Failed to parse NCX TOC: {e}")

    spine_ids = []
    for spine_tuple in book.spine:
        if spine_tuple[1] == 'yes': 
            spine_ids.append(spine_tuple[0])
            
    items = {}
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            items[item.get_id()] = item
            
    for id in spine_ids:
        item = items.get(id, None)
        if item is None:
            continue
            
        html_title, chapter_paragraphs = chap2text_epub(item.get_content(), encoding)
        
        chapter_title = None
        if item.file_name in toc_map:
            chapter_title = toc_map[item.file_name]
        else:
            base_name = os.path.basename(item.file_name)
            for toc_src, toc_title in toc_map.items():
                if os.path.basename(toc_src) == base_name:
                    chapter_title = toc_title
                    break
                    
        if not chapter_title:
            chapter_title = html_title

        book_contents.append({"title": chapter_title, "paragraphs": chapter_paragraphs})
        
    outfile = sourcefile.replace(".epub", ".txt")
    check_for_file(outfile)
    print(f"Exporting {sourcefile} to {outfile} using encoding: {encoding}")
    
    author = book.get_metadata("DC", "creator")[0][0]
    booktitle = book.get_metadata("DC", "title")[0][0]
    
    with open(outfile, "w", encoding='utf-8') as file:
        file.write(f"Title: {booktitle}\n")
        file.write(f"Author: {author}\n\n")
        file.write(f"# Title\n")
        file.write(f"{booktitle}, by {author}\n\n")
        
        for i, chapter in enumerate(book_contents, start=1):
            if chapter["paragraphs"] == [] or chapter["paragraphs"] == ['']:
                continue
            else:
                if chapter["title"] == None:
                    file.write(f"# Part {i}\n")
                else:
                    file.write(f"# {chapter['title']}\n\n")
                for paragraph in chapter["paragraphs"]:
                    clean = re.sub(r'[\s\n]+', ' ', paragraph)
                    clean = re.sub(r'[“”]', '"', clean)
                    clean = re.sub(r'[‘’]', "'", clean)
                    file.write(f"{clean}\n\n")

def get_book(sourcefile, encoding="utf-8"):
    book_contents = []
    book_title = sourcefile
    book_author = "Unknown"
    chapter_titles = []
    
    # Text reader now also respects the user's encoding choice
    with open(sourcefile, "r", encoding=encoding, errors="replace") as file:
        current_chapter = {"title": "blank", "paragraphs": []}
        initialized_first_chapter = False
        lines_skipped = 0
        for line in file:
            if lines_skipped < 2 and (line.startswith("Title") or line.startswith("Author")):
                lines_skipped += 1
                if line.startswith('Title: '):
                    book_title = line.replace('Title: ', '').strip()
                elif line.startswith('Author: '):
                    book_author = line.replace('Author: ', '').strip()
                continue
            
            line = line.strip()
            
            if line.startswith("#"):
                if current_chapter["paragraphs"] or not initialized_first_chapter:
                    if initialized_first_chapter:
                        book_contents.append(current_chapter)
                    current_chapter = {"title": None, "paragraphs": []}
                    initialized_first_chapter = True
                chapter_title = line[1:].strip()
                if any(c.isalnum() for c in chapter_title):
                    current_chapter["title"] = chapter_title
                    chapter_titles.append(current_chapter["title"])
                else:
                    current_chapter["title"] = "blank"
                    chapter_titles.append("blank")
            elif line:
                if not initialized_first_chapter:
                    chapter_titles.append("blank")
                    initialized_first_chapter = True
                if any(char.isalnum() for char in line):
                    sentences = sent_tokenize(line)
                    cleaned_sentences = [s for s in sentences if any(char.isalnum() for char in s)]
                    line = ' '.join(cleaned_sentences)
                    current_chapter["paragraphs"].append(line)
        if current_chapter["paragraphs"]:
            book_contents.append(current_chapter)
    return book_contents, book_title, book_author, chapter_titles

def sort_key(s):
    return int(re.findall(r'\d+', s)[0])

def check_for_file(filename):
    if os.path.isfile(filename):
        print(f"The file '{filename}' already exists.")
        overwrite = input("Do you want to overwrite the file? (y/n): ")
        if overwrite.lower() != 'y':
            print("Exiting without overwriting the file.")
            sys.exit()
        else:
            os.remove(filename)

def append_silence(tempfile, duration=1200):
    audio = AudioSegment.from_file(tempfile)
    silence = AudioSegment.silent(duration)
    combined = audio + silence
    combined.export(tempfile, format="flac")

def read_book(book_contents, speaker, paragraphpause, sentencepause, failed_sentences):
    segments = []
    title_names_to_skip_reading = ['Title', 'blank']
    for i, chapter in enumerate(book_contents, start=1):
        files = []
        partname = f"part{i}.flac"
        print(f"\n\n")
        if os.path.isfile(partname):
            print(f"{partname} exists, skipping to next chapter")
            segments.append(partname)
        else:
            if chapter["title"] in title_names_to_skip_reading:
                print(f"Chapter name: \"{chapter['title']}\" (Skipping title read)")
            else:
                print(f"Chapter name: \"{chapter['title']}\"")
            if chapter["title"] == "":
                chapter["title"] = "blank"
            if chapter["title"] not in title_names_to_skip_reading:
                asyncio.run(parallel_edgespeak([chapter["title"]], [speaker], ["sntnc0.mp3"], failed_sentences))
                append_silence("sntnc0.mp3", 1200)
            for pindex, paragraph in enumerate(tqdm(chapter["paragraphs"], desc=f"Generating audio files: ",unit='pg')):
                ptemp = f"pgraphs{pindex}.flac"
                if os.path.isfile(ptemp):
                    pass
                else:
                    sentences = sent_tokenize(paragraph)
                    filenames = ["sntnc" + str(z + 1) + ".mp3" for z in range(len(sentences))]
                    speakers = [speaker] * len(sentences)
                    
                    # Pass the tracking list into the parallel generator
                    asyncio.run(parallel_edgespeak(sentences, speakers, filenames, failed_sentences))
                    
                    if os.path.exists(filenames[-1]):
                        append_silence(filenames[-1], paragraphpause)
                        
                    sorted_files = sorted(filenames, key=sort_key)
                    if os.path.exists("sntnc0.mp3"):
                        sorted_files.insert(0, "sntnc0.mp3")
                    combined = AudioSegment.empty()
                    for file in sorted_files:
                        if os.path.exists(file): 
                            combined += AudioSegment.from_file(file)
                    combined.export(ptemp, format="flac")
                    for file in sorted_files:
                        if os.path.exists(file):
                            os.remove(file)
                files.append(ptemp)
            
            if files and os.path.exists(files[-1]):
                append_silence(files[-1], 2000)
            
            combined = AudioSegment.empty()
            for file in files:
                if os.path.exists(file):
                    combined += AudioSegment.from_file(file)
            combined.export(partname, format="flac")
            for file in files:
                if os.path.exists(file):
                    os.remove(file)
            segments.append(partname)
    return segments

def generate_metadata(files, author, title, chapter_titles):
    chap = 0
    start_time = 0
    with open("FFMETADATAFILE", "w", encoding='utf-8') as file:
        file.write(";FFMETADATA1\n")
        file.write(f"ARTIST={author}\n")
        file.write(f"ALBUM={title}\n")
        file.write(f"TITLE={title}\n")
        file.write("DESCRIPTION=Made with https://github.com/aedocw/epub2tts-edge\n")
        for file_name in files:
            duration = get_duration(file_name)
            file.write("[CHAPTER]\n")
            file.write("TIMEBASE=1/1000\n")
            file.write(f"START={start_time}\n")
            file.write(f"END={start_time + duration}\n")
            file.write(f"title={chapter_titles[chap]}\n")
            chap += 1
            start_time += duration

def get_duration(file_path):
    audio = AudioSegment.from_file(file_path)
    return len(audio)

def make_audiobook(files, sourcefile, speaker, codec, bitrate, cover_img):
    filelist = "filelist.txt"
    basefile = sourcefile.replace(".txt", "")
    output_intermediate_flac = f"{basefile}_temp.flac"
    output_final = f"{basefile} ({speaker}).mka"
    
    with open(filelist, "w", encoding='utf-8') as f:
        for filename in files:
            filename = filename.replace("'", "'\\''")
            f.write(f"file '{filename}'\n")

    print("Concatenating audio segments...")
    subprocess.run([
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", filelist,
        "-c:a", "flac", "-f", "flac", "-y", output_intermediate_flac
    ])

    print(f"Encoding final file to {output_final} using {codec} at {bitrate}...")
    
    cmd = [
        "ffmpeg",
        "-i", output_intermediate_flac,
        "-i", "FFMETADATAFILE",
    ]

    map_cmd = [
        "-map_metadata", "1",
        "-map", "0:0",
        "-c:a", codec,
        "-b:a", bitrate,
        "-f", "matroska"
    ]

    if cover_img and os.path.isfile(cover_img):
        print(f"Embedding cover art: {cover_img}")
        cmd.extend(["-attach", cover_img])
        map_cmd.extend([
            "-metadata:s:t", "mimetype=image/jpeg"
        ])
        if cover_img.lower().endswith(".png"):
             map_cmd[-1] = "mimetype=image/png"

    cmd.extend(map_cmd)
    cmd.append("-y")
    cmd.append(output_final)

    subprocess.run(cmd)

    if os.path.exists(filelist): os.remove(filelist)
    if os.path.exists("FFMETADATAFILE"): os.remove("FFMETADATAFILE")
    if os.path.exists(output_intermediate_flac): os.remove(output_intermediate_flac)
    for f in files:
        if os.path.exists(f): os.remove(f)

    return output_final

async def run_edgespeak(sentence, speaker, filename, failed_sentences):
    # Native async function: much faster and less resource intensive
    for speakattempt in range(5):
        try:
            communicate = edge_tts.Communicate(sentence, speaker)
            await communicate.save(filename) 
            
            if not os.path.exists(filename) or os.path.getsize(filename) == 0:
                raise Exception("Failed to save file from edge_tts")
            break
        except Exception as e:
            # Non-blocking pause. Doesn't freeze the CPU like time.sleep()
            await asyncio.sleep(2 + speakattempt)
    else:
        print(f"\n⚠️ WARNING: Giving up on sentence '{sentence[:50]}...'. Replacing with silence.")
        
        # Add the failed sentence to our tracking list
        failed_sentences.append(sentence)
        
        try:
            silence = AudioSegment.silent(duration=1000)
            silence.export(filename, format="mp3")
        except Exception as e:
            print(f"Failed to create fallback silence: {e}")

def run_save(communicate, filename):
    asyncio.run(communicate.save(filename))

async def parallel_edgespeak(sentences, speakers, filenames, failed_sentences):
    semaphore = asyncio.Semaphore(5) # Keeps Azure from IP-banning you
    
    async def bounded_edgespeak(sentence, speaker, filename):
        async with semaphore:
            sentence = re.sub(r'[!]+', '!', sentence)
            sentence = re.sub(r'[?]+', '?', sentence)
            await run_edgespeak(sentence, speaker, filename, failed_sentences)

    tasks = [
        bounded_edgespeak(sentence, speaker, filename) 
        for sentence, speaker, filename in zip(sentences, speakers, filenames)
    ]
    await asyncio.gather(*tasks)

def main():
    parser = argparse.ArgumentParser(prog="epub2tts-edge")
    parser.add_argument("sourcefile", type=str, help="The epub or text file to process")
    parser.add_argument("--speaker", type=str, default="en-US-AndrewNeural")
    parser.add_argument("--cover", type=str, help="jpg image to use for cover")
    parser.add_argument("--sentencepause", type=int, default=1200)
    parser.add_argument("--paragraphpause", type=int, default=1200)
    parser.add_argument("--codec", type=str, default="libopus", help="Audio codec (default: libopus)")
    parser.add_argument("--bitrate", type=str, default="35k", help="Bitrate (default: 35k)")
    parser.add_argument("--encoding", type=str, default="utf-8", help="Force text encoding (default: utf-8). Useful for fixing Mojibake.")

    args = parser.parse_args()
    print(args)

    ensure_punkt()

    if args.sourcefile.endswith(".epub"):
        book = epub.read_epub(args.sourcefile)
        export(book, args.sourcefile, encoding=args.encoding)
        auto_cover = args.sourcefile.replace(".epub", ".png")
        if args.cover is None and os.path.exists(auto_cover):
             args.cover = auto_cover
        exit()

    book_contents, book_title, book_author, chapter_titles = get_book(args.sourcefile, encoding=args.encoding)
    
    # Initialize the tracking list
    failed_sentences = []
    
    files = read_book(book_contents, args.speaker, args.paragraphpause, args.sentencepause, failed_sentences)
    generate_metadata(files, book_author, book_title, chapter_titles)
    
    final_file = make_audiobook(files, args.sourcefile, args.speaker, args.codec, args.bitrate, args.cover)
    print(f"Done! File saved as: {final_file}")

    # Courtesy Summary printed at the very end
    if failed_sentences:
        print("\n" + "="*80)
        print("⚠️  CONVERSION COMPLETED, BUT SOME SENTENCES FAILED ⚠️")
        print("The following text triggered engine errors and was replaced with silence:")
        print("-" * 80)
        for i, sentence in enumerate(failed_sentences, 1):
            print(f"{i}. {sentence}")
        print("="*80 + "\n")
    else:
        print("\n✅ CONVERSION 100% SUCCESSFUL! No sentences were dropped.\n")

if __name__ == "__main__":
    main()
