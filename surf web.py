from asyncio import Handle
from audioop import error
from os import link
import re
import sys
import time
import math
import unicodedata
from urllib import request
import urllib2
import urlparse
import optparse
import hashli
from cgi import escape
from traceback import format_exc
from Queue import Queue, Empty as QueueEmpty
from bs4 import BeautifulSoup


class Link (object):
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
class Crawler(object):
    def __init__(self, root, depth_limit, confine=None, exclude=[], locked=True, filter_seen=True):
        self.root = root
        self.host = urlparse.urlparse(root)[1]
        self.depth_limit = depth_limit # Max depth (number of hops from root)
        self.locked = locked           # Limit search to a single host?
        self.confine_prefix=confine    # Limit search to this prefix
        self.exclude_prefixes=exclude; # URL prefixes NOT to visit
        self.urls_seen = set()          # Used to avoid putting duplicates in queue
        self.urls_remembered = set()    # For reporting to user
        self.visited_links= set()       # Used to avoid re-processing a page
        self.links_remembered = set()   # For reporting to user
        self.num_links = 0              # Links found (and not excluded by filters)
        self.num_followed = 0           # Links followed.  
        self.pre_visit_filters=[self._prefix_ok,
                                self._exclude_ok,
                                self._not_visited,
                                self._same_host]
    if filter_seen:
            self.out_url_filters=[self._prefix_ok,
                                     self._same_host]
    else :
            self.out_url_filters=[]
    def _pre_visit_url_condense(self, url):
        base, frag = urlparse.urldefrag(url)
        return base
    def _prefix_ok(self, url):
        """Pass if the URL has the correct prefix, or none is specified"""
        return (self.confine_prefix is None  or
                url.startswith(self.confine_prefix))
    def _exclude_ok(self, url):
        """Pass if the URL does not match any exclude patterns"""
        prefixes_ok = [ not url.startswith(p) for p in self.exclude_prefixes]
        return all(prefixes_ok)
    def _not_visited(self, url):
        """Pass if the URL has not already been visited"""
        return (url not in self.visited_links)
    def _same_host(self, url):
        """Pass if the URL is on the same host as the root URL"""
        try:
            host = urlparse.urlparse(url)[1]
            return re.match(".*%s" % self.host, host) 
        except Exception.e:
            print >> sys.stderr, "ERROR: Can't process url '%s' (%s)" % (url, e)
            return False
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
            if [] == do_not_follow:
                try:
                    self.visited_links.add(this_url)
                    self.num_followed += 1
                    page = Fetcher(this_url)
                    page.fetch()
                    for link_url in [self._pre_visit_url_condense(l) for l in page.out_links()]:
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
class OpaqueDataException (Exception):
    def __init__(self, message, mimetype, url):
        Exception.__init__(self, message)
        self.mimetype=mimetype
        self.url=url
class Fetcher(object):
    def __init__(self, url):
        self.url = url
        self.out_url = []
    def __getitem__(self, x):
        return self.out_url[x]
    def out_link__(self):
        return self.out_url
    def _addHeaders(self, request):
        request.add_header("User-Agent", AGENT)
    def _open(self):
        url = self.url
        try:
                    request = urllib2.Request(url)
                    handle = urllib2.build_opener()
        except IOError:
                    return None
        return (request, handle)
    def fetch(self):
        request, handle = self._open()
        if handle: 
            try:
                data=handle.open(request)
                mime_type=data.info().gettype()
                url=data.geturl()
                if mime_type != "text/html":
                    raise OpaqueDataException("Not interested in files of type %s" % mime_type,mime_type, url)
                content = unicodedata(data.read(), "utf-8",
                        errors="replace")
                soup = BeautifulSoup(content)
                tags = soup('a')
            except urllib2.HTTPError. error:
                if error.code == 404:
                    print >> sys.stderr, "ERROR: %s -> %s" % (error, error.url)
                else:
                    print >> sys.stderr, "ERROR: %s" % error
                tags = []
            except urllib2.URLError. error:
                print >> sys.stderr, "ERROR: %s" % error
                tags = []
            except OpaqueDataException. error:
                print >>sys.stderr, "Skipping %s, has type %s" % (error.url, error.mimetype)
                tags = []
            for tag in tags:
                href = tag.get("href")
                if href is not None:
                    url = urlparse.urljoin(self.url, escape(href))
                    if url not in self:
                      self.out_url.append(url)
    def  getLinks(url):
        page = Fetcher(url)
        page.fetch()
    j = 1
    for i, url in enumerate(page):
            if url.find("http")>=0:
                    print >> "%d. %s " % (j, url)
                    j = j + 1
    def parse_options():
            parser = optparse.OptionParser()
            parser.add_option("-q", "--quiet",
                action="store_true", default=False, dest="quiet",
                    help="Enable quiet mode")
            parser.add_option("-l", "--links",
                action="store_true", default=False, dest="links",
                    help="Get links for specified url only")    
            parser.add_option("-d", "--depth",
                action="store", type="int", default=30, dest="depth_limit",
                    help="Maximum depth to traverse")
            parser.add_option("-c", "--confine",
                action="store", type="string", dest="confine",
                    help="Confine crawl to specified prefix")
            parser.add_option("-x")
    import (
	"encoding/xml"
	"io/ioutil"
	"os"
	"fmt"
)

type Urlset struct {
	XMLUrlSet xml.Name `xml:"urlset"`
	Urls   []Url   `xml:"url"`	
}

type Url struct {
	Url xml.Name `xml:"url"`
	Loc string `xml:"loc"`
}

func ReadSiteMap(sitemap string) (urlSet Urlset){
	xmlFile, err := os.Open(sitemap)
	if err != nil {
		fmt.Println(err)
	}

	fmt.Println("Successfully Opened users.xml")
	defer xmlFile.Close()
	byteValue, _ := ioutil.ReadAll(xmlFile)
	xml.Unmarshal(byteValue, &urlSet)

	return 
}
def kmeans(init_centes, init_labels, X, n_cluster):
  centers = init_centes
  labels = init_labels
  times = 0
  while True:
    labels = kmeans_predict_labels(X, centers)
    kmeans_visualize(X, centers, labels, n_cluster, 'Assigned label for data at time = ' + str(times + 1))
    new_centers = kmeans_update_centers(X, labels, n_cluster)
    if kmeans_has_converged(centers, new_centers):
      break
    centers = new_centers
    kmeans_visualize(X, centers, labels, n_cluster, 'Update center possition at time = ' + str(times + 1))
    times += 1
  return (centers, labels, times)
  init_centers = kmeans_init_centers(X, n_cluster)
print(init_centers) # In ra tọa độ khởi tạo ban đầu của các tâm cụm
init_labels = np.zeros(X.shape[0])
kmeans_visualize(X, init_centers, init_labels, n_cluster, 'Init centers in the first run. Assigned all data as cluster 0')
centers, labels, times = kmeans(init_centers, init_labels, X, n_cluster)
 
print('Done! Kmeans has converged after', times, 'times')



        