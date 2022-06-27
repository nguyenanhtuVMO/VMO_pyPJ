import re
import sys
import time
import math
import unicodedata
from os import link



class Link(ob):
       def __init__(self, src, dst, link_type):
        self.src = src
        self.dst = dst
        self.link_type = link_type
def __hash__(self):
        return hash((self.src, self.dst, self.link_type))
def __eq__(self, other):
        return (self.src == other.src and
                self.dst == other.dst and
                self.link_type == other.link_type)
def __str__(self):
        return self.src + " -> " + self.dst

def _crawl(self):
        q = Queue
        q.put((self.root, 0))
        q.put((self.root, 0))
        while not q.empty():
            this_url, depth = q.get()
            #Non-URL-specific filter: Discard anything over depth limit
            if depth > self.depth_limit:
                continue
            do_not_follow = [f for f in self.pre_visit_filters if not f(this_url)]
            #Special-case depth 0 (starting URL)
            if depth == 0 and [] != do_not_follow:
                print >> sys.stderr, "Whoops! Starting URL %s rejected by the following filters:", do_not_follow
            if[] == dont_fl:
                try:
                    self.visited_Link.add(this_url)
                    self.num_fl += 1
                    page = Fetcher(this_url)
                    page.fetch()
                    for link_url in [self._pre_visit_yrl_condense(1) for 1 in page.out_links()]:
                         if link_url not in self.urls_seen:
                            q.put((link_url, depth+1))
                            self.urls_seen.add(link_url)
                            do_not_remember = [f for f in self.out_url_filters if not f(link_url)]
                         if [] == do_not_remember:
                                self.num_links += 1
                                self.urls_remembered.add(link_url)
                                link = Link(this_url, link_url, "href")
                                if link not in self.links_remembered:
                                    self.links_remembered.add(link)
                except Exception. e:
                    print >>sys.stderr, "ERROR: Can't process url '%s' (%s)" % (this_url, e)