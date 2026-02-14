"""
SEC Filing Downloader - FIXED VERSION
Gets the full text filing, not the XBRL viewer
"""

import requests
from bs4 import BeautifulSoup
import time
import json
from pathlib import Path

def download_sec_filing(company_cik, company_name, filing_type="10-K", num_filings=1):
    """Download SEC filings from EDGAR - gets full text version"""
    
    base_url = "https://www.sec.gov"
    headers = {"User-Agent": "etuyishi@mail.yu.edu"}
    
    print("\n" + "="*60)
    print(f"Downloading {company_name} {filing_type} filings...")
    print("="*60)
    
    # Search for filings
    search_url = f"{base_url}/cgi-bin/browse-edgar"
    params = {
        "action": "getcompany",
        "CIK": company_cik,
        "type": filing_type,
        "dateb": "",
        "owner": "exclude",
        "count": num_filings * 2
    }
    
    try:
        response = requests.get(search_url, params=params, headers=headers)
        response.raise_for_status()
    except Exception as e:
        print(f"ERROR searching: {e}")
        return None
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find filing links and accession numbers
    filing_data = []
    
    for row in soup.find_all('tr'):
        cells = row.find_all('td')
        if len(cells) >= 4 and filing_type in cells[0].text:
            # Get the filing date
            filing_date = cells[3].text.strip()
            
            # Get the accession number (needed for full text link)
            link = cells[1].find('a')
            if link and 'Archives' in link['href']:
                # Extract accession number from URL
                # Format: /Archives/edgar/data/320193/000032019324000123/index.html
                parts = link['href'].split('/')
                if len(parts) >= 5:
                    accession_num = parts[4].replace('-', '')  # Remove hyphens
                    filing_data.append({
                        'date': filing_date,
                        'accession': accession_num,
                        'cik': company_cik
                    })
    
    print(f"Found {len(filing_data)} filings")
    
    if not filing_data:
        return None
    
    # Download filings
    filings = []
    for i, filing_info in enumerate(filing_data[:num_filings]):
        date = filing_info['date']
        accession = filing_info['accession']
        cik = filing_info['cik']
        
        print(f"\nDownloading filing {i+1}/{num_filings} (Date: {date})...")
        
        try:
            # Construct the FULL TEXT filing URL
            # Format: https://www.sec.gov/cgi-bin/viewer?action=view&cik=320193&accession_number=0000320193-24-000123&xbrl_type=v
            # OR use the -index.htm file which contains full text
            
            # Try to get the full submission text file
            # Format: /Archives/edgar/data/CIK/ACCESSION/ACCESSION.txt
            cik_no_zeros = cik.lstrip('0')  # Remove leading zeros
            
            # Method 1: Try the .txt file (full submission)
            txt_url = f"{base_url}/Archives/edgar/data/{cik_no_zeros}/{accession}/{accession}.txt"
            
            print(f"  Trying: {txt_url}")
            doc_response = requests.get(txt_url, headers=headers)
            
            if doc_response.status_code == 200:
                content = doc_response.text
                
                # The .txt file contains the full submission with tags
                # Extract just the main document (between <DOCUMENT> tags)
                if '<DOCUMENT>' in content:
                    # Find the main 10-K document
                    doc_start = content.find('<TYPE>10-K')
                    if doc_start == -1:
                        doc_start = content.find('<TYPE>' + filing_type)
                    
                    if doc_start > -1:
                        # Find the actual text/html content
                        text_start = content.find('<TEXT>', doc_start)
                        text_end = content.find('</TEXT>', text_start)
                        
                        if text_start > -1 and text_end > -1:
                            content = content[text_start+6:text_end]
                
                # Clean HTML if present
                if '<html' in content.lower() or '<HTML' in content:
                    soup2 = BeautifulSoup(content, 'html.parser')
                    # Remove script, style, and table tags (tables are often formatting)
                    for tag in soup2(["script", "style"]):
                        tag.decompose()
                    content = soup2.get_text()
                
                # Clean whitespace
                lines = (line.strip() for line in content.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                final_content = ' '.join(chunk for chunk in chunks if chunk)
                
                # Verify we got real content (not just error message)
                if len(final_content) < 1000:
                    print(f"  WARNING: Content too short ({len(final_content)} chars)")
                    print(f"  Content preview: {final_content[:200]}")
                    continue
                
                print(f"  Downloaded {len(final_content):,} characters")
                
                filings.append({
                    'type': filing_type,
                    'date': date,
                    'url': txt_url,
                    'content': final_content
                })
                
            else:
                print(f"  ERROR: Could not fetch document (status {doc_response.status_code})")
                continue
            
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    if not filings:
        print("ERROR: No filings successfully downloaded")
        return None
    
    return {
        'company_name': company_name,
        'cik': company_cik,
        'filings': filings
    }


def main():
    """Download filings for multiple companies"""
    
    companies = {
        'Apple': '0000320193',
        'Microsoft': '0000789019',
        'Tesla': '0001318605',
        'Amazon': '0001018724',
        'Google': '0001652044',
    }
    
    output_dir = Path('data/raw/sec_filings')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("SEC FILING DOWNLOADER - FIXED VERSION")
    print("="*60)
    print(f"\nDownloading 10-K filings for {len(companies)} companies")
    print("This will take 5-10 minutes...")
    print("Getting FULL TEXT versions (not XBRL viewer)\n")
    
    successful = 0
    
    for company_name, cik in companies.items():
        data = download_sec_filing(cik, company_name, '10-K', 1)
        
        if data:
            filename = f"{company_name.lower()}_filings.json"
            filepath = output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            total_chars = sum(len(f['content']) for f in data['filings'])
            print(f"\nSaved: {filepath}")
            print(f"  Company: {company_name}")
            print(f"  Filings: {len(data['filings'])}")
            print(f"  Total chars: {total_chars:,}")
            
            # Verify it's real content
            if total_chars < 10000:
                print(f"  WARNING: File seems too small!")
            else:
                print(f"  SUCCESS: Got real filing content")
            
            successful += 1
        
        time.sleep(1)
    
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print(f"\nDownloaded: {successful}/{len(companies)} companies")
    
    if successful > 0:
        print("\nNext step: python3 build_rag.py")
    else:
        print("\nERROR: No filings downloaded successfully")
        print("The SEC might be blocking requests or files have moved")


if __name__ == "__main__":
    main()