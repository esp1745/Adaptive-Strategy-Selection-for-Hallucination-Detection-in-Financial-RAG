"""
SEC Filing Downloader v2 - Uses SEC EDGAR API
Downloads actual 10-K filing content (not XBRL viewer)
"""

import requests
from bs4 import BeautifulSoup
import time
import json
import re
from pathlib import Path


def get_filing_index(cik: str, filing_type: str = "10-K", headers: dict = None):
    """Get list of filings from SEC EDGAR API"""
    
    # SEC EDGAR submissions API
    cik_padded = cik.lstrip('0').zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    
    print(f"  Fetching filing index from: {url}")
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"  ERROR: Could not fetch submissions (status {response.status_code})")
        return []
    
    data = response.json()
    filings = data.get('filings', {}).get('recent', {})
    
    # Find 10-K filings
    results = []
    forms = filings.get('form', [])
    accessions = filings.get('accessionNumber', [])
    dates = filings.get('filingDate', [])
    primary_docs = filings.get('primaryDocument', [])
    
    for i, form in enumerate(forms):
        if form == filing_type:
            results.append({
                'accession': accessions[i].replace('-', ''),
                'accession_formatted': accessions[i],
                'date': dates[i],
                'primary_doc': primary_docs[i]
            })
            if len(results) >= 2:  # Get last 2 filings
                break
    
    return results


def download_filing_content(cik: str, accession: str, primary_doc: str, headers: dict):
    """Download the actual filing content"""
    
    cik_no_zeros = cik.lstrip('0')
    
    # Try multiple methods to get the filing content
    content = None
    
    # Method 1: Try the primary document (usually the main HTML file)
    doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_no_zeros}/{accession}/{primary_doc}"
    print(f"  Trying primary doc: {doc_url}")
    
    response = requests.get(doc_url, headers=headers)
    if response.status_code == 200:
        content = response.text
        
        # Parse HTML content
        if '<html' in content.lower():
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove unwanted tags
            for tag in soup(['script', 'style', 'meta', 'link']):
                tag.decompose()
            
            # Get text
            content = soup.get_text(separator=' ')
    
    # Method 2: If primary doc didn't work, try the .txt full submission
    if not content or len(content) < 5000:
        txt_url = f"https://www.sec.gov/Archives/edgar/data/{cik_no_zeros}/{accession}/{accession}.txt"
        print(f"  Trying full submission: {txt_url}")
        
        response = requests.get(txt_url, headers=headers)
        if response.status_code == 200:
            raw_content = response.text
            
            # Extract the main document from the full submission
            # Look for <TYPE>10-K and extract content between <TEXT> tags
            if '<DOCUMENT>' in raw_content:
                # Find 10-K section
                pattern = r'<TYPE>10-K.*?<TEXT>(.*?)</TEXT>'
                match = re.search(pattern, raw_content, re.DOTALL | re.IGNORECASE)
                
                if match:
                    content = match.group(1)
                    
                    # Clean HTML
                    if '<html' in content.lower():
                        soup = BeautifulSoup(content, 'html.parser')
                        for tag in soup(['script', 'style']):
                            tag.decompose()
                        content = soup.get_text(separator=' ')
    
    if not content:
        return None
    
    # Clean whitespace
    content = re.sub(r'\s+', ' ', content)
    content = content.strip()
    
    return content


def download_company_filings(cik: str, company_name: str, filing_type: str = "10-K"):
    """Download filings for a company"""
    
    headers = {
        "User-Agent": "YeshivaUniversity etuyishi@mail.yu.edu",
        "Accept-Encoding": "gzip, deflate"
    }
    
    print(f"\n{'='*60}")
    print(f"Downloading {company_name} {filing_type} filings...")
    print("="*60)
    
    # Get filing index
    filings_info = get_filing_index(cik, filing_type, headers)
    
    if not filings_info:
        print(f"  No {filing_type} filings found")
        return None
    
    print(f"  Found {len(filings_info)} {filing_type} filings")
    
    filings = []
    for filing_info in filings_info[:1]:  # Get the most recent one
        print(f"\n  Processing filing from {filing_info['date']}...")
        
        content = download_filing_content(
            cik, 
            filing_info['accession'],
            filing_info['primary_doc'],
            headers
        )
        
        if content and len(content) > 5000:
            print(f"  SUCCESS: Downloaded {len(content):,} characters")
            
            filings.append({
                'type': filing_type,
                'date': filing_info['date'],
                'accession': filing_info['accession_formatted'],
                'content': content
            })
        else:
            print(f"  WARNING: Content too short or empty")
        
        time.sleep(0.5)  # Rate limiting
    
    if not filings:
        return None
    
    return {
        'company_name': company_name,
        'cik': cik,
        'filings': filings
    }


def main():
    """Download 10-K filings for target companies"""
    
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
    print("SEC FILING DOWNLOADER v2")
    print("="*60)
    print(f"Downloading 10-K filings for {len(companies)} companies")
    print("Using SEC EDGAR API for reliable downloads\n")
    
    successful = 0
    
    for company_name, cik in companies.items():
        data = download_company_filings(cik, company_name, '10-K')
        
        if data and data['filings']:
            filename = f"{company_name.lower()}_filings.json"
            filepath = output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            total_chars = sum(len(f['content']) for f in data['filings'])
            print(f"\n  Saved: {filepath}")
            print(f"    Total content: {total_chars:,} characters")
            
            # Show a preview
            preview = data['filings'][0]['content'][:200]
            print(f"    Preview: {preview}...")
            
            successful += 1
        else:
            print(f"\n  FAILED: Could not download {company_name} filings")
        
        time.sleep(1)  # Be nice to SEC servers
    
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print(f"Successfully downloaded: {successful}/{len(companies)} companies")
    
    if successful > 0:
        print("\nNext step: python3 build_rag.py")
    else:
        print("\nNo filings downloaded. Check your internet connection.")


if __name__ == "__main__":
    main()
