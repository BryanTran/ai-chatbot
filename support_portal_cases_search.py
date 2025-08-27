import requests
from bs4 import BeautifulSoup
import urllib.parse
import time
from langchain_core.tools import tool

def get_case_descriptions(search_term: str) -> str:
    """
    Scrapes case descriptions from the support site for the given search term,
    and returns a single combined string of all descriptions found.
    """
    session = requests.Session()

    # Step 1: Login
    login_url = 'https://support.multitech.com/support/login.html'
    dashboard_url = 'https://support.multitech.com/support/dashboard.html'
    payload = {
        'pas7urd': 'btran@multitech.com',
        'u32namb4': 'trong2',
        'login': 'Login'
    }

    login_response = session.post(login_url, data=payload)
    if 'Dashboard' not in login_response.text:
        print("[!] Login failed.")
        return ""
    print("[+] Login successful.")

    # Step 2: Prepare search payload
    search_payload = {
        'caseAJAX': 'true',
        # 'tabselect': 'allclosed',
        'tabselect': 'casesclosed',
        'filter': 'all',
        'caseorder': 'updated',
        'page': 1,
        'row_count': 10,
        'searchinput': search_term
    }

    # Step 3: Begin scraping
    all_descriptions = []

    while True:
        response = session.post(dashboard_url, data=search_payload)
        soup = BeautifulSoup(response.text, "html.parser")

        case_links = []
        for a in soup.find_all("a", href=True):
            if "case.html?action=view&id=" in a["href"]:
                full_url = urllib.parse.urljoin(dashboard_url, a["href"])
                if full_url not in case_links:
                    case_links.append(full_url)

        print(f"[+] Found {len(case_links)} case links on page {search_payload['page']}.")

        if not case_links:
            print("[!] No case links found. Stopping.")
            break

        for case_url in case_links:
            print(f"[*] Visiting case: {case_url}")
            case_resp = session.get(case_url)
            case_soup = BeautifulSoup(case_resp.text, "html.parser")

            description_text = None
            for row_class in ['unreadworkitem', 'readworkitem']:
                for tr in case_soup.find_all("tr", class_=row_class):
                    tds = tr.find_all("td")
                    if tds and tds[0].get_text(strip=True) == "Description:":
                        content_td = tds[1] if len(tds) > 1 else None
                        if content_td:
                            div = content_td.find("div")
                            if div:
                                description_text = div.get_text(separator="\n", strip=True)
                                print(f"[+] Found description in case: {case_url}")
                                break
                if description_text:
                    break

            if description_text:
                combined_entry = f"URL: {case_url}\n{description_text}\n{'=' * 80}"
                all_descriptions.append(combined_entry)
            else:
                print(f"[!] No description found for case: {case_url}")

            time.sleep(0.5)

        # Step 4: Check if there's a next page
        pagination = soup.find("a", string=lambda s: s and s.startswith("Last("))
        if pagination:
            try:
                last_page = int(pagination.text.strip("Last() "))
                if search_payload['page'] >= last_page:
                    print("[*] Reached last page.")
                    break
                else:
                    search_payload['page'] += 1
            except Exception as e:
                print(f"[!] Error parsing last page number: {e}")
                break
        else:
            print("[*] No pagination found. Assuming single page.")
            break

    print(f"[✓] Total descriptions collected: {len(all_descriptions)}")
    text = "\n\n".join(all_descriptions)

    print(text)

    return text

@tool
def get_case_descriptions_wrapper(search_query: str) -> str:
    """
    Wrapper to make get_case_descriptions compatible with LangChain Tools.
    """
    print(f"[Tool] Searching support cases for: {search_query}")
    return get_case_descriptions(search_term=search_query)


