// Command ask sends a question to the RAG API and prints the answer.
//
//	go run ./cmd/ask -url http://localhost:8000 "Which TCP port does router control use?"
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/Snehachowdary56/rag-pipeline-optimization/go/ragclient"
)

func main() {
	url := flag.String("url", "http://localhost:8000", "RAG API base URL")
	k := flag.Int("k", 3, "number of chunks to retrieve")
	flag.Parse()
	question := strings.Join(flag.Args(), " ")
	if question == "" {
		fmt.Fprintln(os.Stderr, "usage: ask [-url URL] [-k N] question")
		os.Exit(2)
	}
	a, err := ragclient.New(*url).Ask(context.Background(), question, *k)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	fmt.Println(a.Answer)
	fmt.Printf("sources: %s | grounded: %v | cached: %v | %.2f ms\n", strings.Join(a.Citations, ", "), a.Grounded, a.Cached, a.LatencyMS)
}
