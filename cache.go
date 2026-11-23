package main

// cache.go — tiny TTL + LRU cache for identical requests (no external deps).

import (
	"container/list"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sync"
	"time"
)

type cacheEntry struct {
	key     string
	value   []byte
	expires time.Time
}

type LRU struct {
	mu    sync.Mutex
	cap   int
	ttl   time.Duration
	items map[string]*list.Element
	order *list.List
	hits  int64
	miss  int64
}

func NewLRU(capacity int, ttl time.Duration) *LRU {
	if capacity <= 0 {
		capacity = 256
	}
	return &LRU{
		cap:   capacity,
		ttl:   ttl,
		items: make(map[string]*list.Element, capacity),
		order: list.New(),
	}
}

func (c *LRU) Get(key string) ([]byte, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	el, ok := c.items[key]
	if !ok {
		c.miss++
		return nil, false
	}
	e := el.Value.(*cacheEntry)
	if time.Now().After(e.expires) {
		c.order.Remove(el)
		delete(c.items, key)
		c.miss++
		return nil, false
	}
	c.order.MoveToFront(el)
	c.hits++
	return e.value, true
}

func (c *LRU) Put(key string, value []byte) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if el, ok := c.items[key]; ok {
		e := el.Value.(*cacheEntry)
		e.value = value
		e.expires = time.Now().Add(c.ttl)
		c.order.MoveToFront(el)
		return
	}
	el := c.order.PushFront(&cacheEntry{key: key, value: value, expires: time.Now().Add(c.ttl)})
	c.items[key] = el
	for c.order.Len() > c.cap {
		last := c.order.Back()
		if last == nil {
			break
		}
		c.order.Remove(last)
		delete(c.items, last.Value.(*cacheEntry).key)
	}
}

func (c *LRU) Stats() (hits, miss int64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.hits, c.miss
}

// cacheKey hashes everything that changes the answer.
func cacheKey(v any) string {
	b, _ := json.Marshal(v)
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}
