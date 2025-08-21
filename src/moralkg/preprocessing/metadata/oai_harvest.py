"""
Inspired by:
https://github.com/thoppe/The-Pile-PhilPapers/blob/master/pyoaiharvest.py
"""

import re
import time
from typing import Optional
import requests
import xml.dom.pulldom
import codecs

from moralkg.logging import get_logger

class Harvester:
    """Simple OAI-PMH ListRecords harvester specialized for PhilPapers/PhilArchive.

    Writes a minimal XML repository containing only <record> elements so that
    downstream parsers can process files without network access.
    """

    def __init__(self, user_agent: str = "OAIHarvester/2.0", max_retries: int = 5):
        self.user_agent = user_agent
        self.max_retries = max_retries
        self._recoveries = 0
        self._logger = get_logger(__name__)

    def _get(self, url: str) -> Optional[str]:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html",
            "Accept-Encoding": "compress, deflate",
        }
        try:
            r = requests.get(url, headers=headers, timeout=60)
        except Exception:
            r = None
        if not r or r.status_code != 200:
            self._recoveries += 1
            self._logger.warning(
                "HTTP error while fetching %s (status=%s); retry %d/%d",
                url,
                getattr(r, "status_code", "ERR"),
                self._recoveries,
                self.max_retries,
            )
            if self._recoveries > self.max_retries:
                return None
            time.sleep(5 * self._recoveries)
            return self._get(url)
        return r.text

    def harvest(
        self,
        server_url: str,
        output_path: str,
        from_date: Optional[str] = None,
        until_date: Optional[str] = None,
        metadata_prefix: str = "oai_dc",
        set_name: Optional[str] = None,
    ) -> int:
        """Harvest records and write them to an XML file.

        Returns the number of <record> elements written.
        """
        server = server_url if server_url.startswith("http") else f"http://{server_url}"

        params = []
        if set_name:
            params.append(f"set={set_name}")
        if from_date:
            params.append(f"from={from_date}")
        if until_date:
            params.append(f"until={until_date}")
        params.append(f"metadataPrefix={metadata_prefix}")
        query = "&".join(params)

        self._logger.info("Using url:%s?ListRecords&%s", server, query)

        # Open in binary and wrap with UTF-8 writer so xml.dom can write unicode
        ofile = codecs.lookup("utf-8")[-1](open(output_path, "wb"))
        ofile.write(
            '<repository xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/" '
            ' xmlns:dc="http://purl.org/dc/elements/1.1/" '
            ' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n'
        )

        record_count = 0
        data = self._get(f"{server}?verb=ListRecords&{query}")

        while data:
            events = xml.dom.pulldom.parseString(data)
            for event, node in events:
                if event == "START_ELEMENT" and node.tagName == "record":
                    events.expandNode(node)
                    node.writexml(ofile)
                    record_count += 1

            mo = re.search("<resumptionToken[^>]*>(.*)</resumptionToken>", data)
            if not mo:
                break
            token = mo.group(1)
            data = self._get(f"{server}?verb=ListRecords&resumptionToken={token}")

        ofile.write("\n</repository>\n")
        ofile.close()

        self._logger.info("Wrote out %d records", record_count)
        return record_count


def harvest_metadata(
    server_url: str,
    output_path: str,
    from_date: Optional[str] = None,
    until_date: Optional[str] = None,
    metadata_prefix: str = "oai_dc",
    set_name: Optional[str] = None,
) -> int:
    """Functional wrapper around OAIHarvester.harvest for quick scripts."""
    return Harvester().harvest(
        server_url=server_url,
        output_path=output_path,
        from_date=from_date,
        until_date=until_date,
        metadata_prefix=metadata_prefix,
        set_name=set_name,
    )
