// Package ragclient is a small Go client for the RAG pipeline REST API.
package ragclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// Answer is the JSON returned by POST /ask.
type Answer struct {
	Question  string   `json:"question"`
	Answer    string   `json:"answer"`
	Citations []string `json:"citations"`
	Grounded  bool     `json:"grounded"`
	Cached    bool     `json:"cached"`
	LatencyMS float64  `json:"latency_ms"`
}

// Client calls the API at BaseURL.
type Client struct {
	BaseURL string
	HTTP    *http.Client
}

// New returns a client with a 10 second timeout.
func New(baseURL string) *Client {
	return &Client{BaseURL: baseURL, HTTP: &http.Client{Timeout: 10 * time.Second}}
}

// Ask sends a question and returns the grounded answer with its citations.
func (c *Client) Ask(ctx context.Context, question string, k int) (*Answer, error) {
	body, err := json.Marshal(map[string]any{"question": question, "k": k})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/ask", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, fmt.Errorf("calling API: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API returned HTTP %d", resp.StatusCode)
	}
	var a Answer
	if err := json.NewDecoder(resp.Body).Decode(&a); err != nil {
		return nil, fmt.Errorf("decoding answer: %w", err)
	}
	return &a, nil
}
