// memory-graph-server serves Obsidian memory-graph static files for Quest/WebXR VR.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"
)

var (
	listenAddr = flag.String("addr", "0.0.0.0:8765", "listen address host:port")
	rootDir    = flag.String("root", "", "static file root (default: HERMES_MEMORY_GRAPH_ROOT or ./output)")
)

type healthResponse struct {
	OK    bool   `json:"ok"`
	Addr  string `json:"addr"`
	Root  string `json:"root"`
	Build string `json:"build"`
}

func resolveRoot() (string, error) {
	if *rootDir != "" {
		return filepath.Clean(*rootDir), nil
	}
	if env := os.Getenv("HERMES_MEMORY_GRAPH_ROOT"); env != "" {
		return filepath.Clean(env), nil
	}
	return filepath.Clean("output"), nil
}

func healthHandler(w http.ResponseWriter, root string) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(healthResponse{
		OK:    true,
		Addr:  *listenAddr,
		Root:  root,
		Build: "go",
	})
}

func main() {
	flag.Parse()

	root, err := resolveRoot()
	if err != nil {
		log.Fatalf("resolve root: %v", err)
	}

	info, err := os.Stat(root)
	if err != nil {
		log.Fatalf("root %q: %v", root, err)
	}
	if !info.IsDir() {
		log.Fatalf("root %q is not a directory", root)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		healthHandler(w, root)
	})
	mux.Handle("/", http.FileServer(http.Dir(root)))

	srv := &http.Server{
		Addr:              *listenAddr,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}

	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		sig := <-sigCh
		log.Printf("signal %v — shutting down", sig)
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := srv.Shutdown(ctx); err != nil {
			log.Printf("shutdown: %v", err)
		}
	}()

	log.Printf("memory-graph-server listening on %s root=%s", *listenAddr, root)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("listen: %v", err)
	}
}
