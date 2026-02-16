import regex as re

class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        
        # Create reverse vocabulary mapping (bytes -> token_id)
        self.reverse_vocab = {bytes_value: token_id for token_id, bytes_value in vocab.items()}
        
        # Encode special tokens to bytes and sort by length (longest first) for proper matching
        self.special_token_bytes = []
        for special_token in self.special_tokens:
            special_bytes = special_token.encode("utf-8")
            # Find the token ID for this special token
            if special_bytes in self.reverse_vocab:
                self.special_token_bytes.append((special_bytes, self.reverse_vocab[special_bytes]))
        
        # Sort by length descending to match longest tokens first
        self.special_token_bytes.sort(key=lambda x: len(x[0]), reverse=True)
        
        # Pre-tokenization pattern (same as training)
        self.pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    def _apply_bpe_merges(self, word):
        """Apply BPE merges to a word (list of byte tokens)."""
        # Start with individual bytes
        for merge_pair in self.merges:
            i = 0
            while i < len(word) - 1:
                if word[i] == merge_pair[0] and word[i + 1] == merge_pair[1]:
                    # Merge the pair
                    merged_token = merge_pair[0] + merge_pair[1]
                    word[i] = merged_token
                    word.pop(i + 1)
                else:
                    i += 1
        return word
    
    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        
        # Handle special tokens first
        # Find all special token positions
        special_positions = []
        text_positions = list(range(len(text)))
        
        for special_bytes, token_id in self.special_token_bytes:
            special_str = special_bytes.decode("utf-8", errors="ignore")
            start = 0
            while True:
                pos = text.find(special_str, start)
                if pos == -1:
                    break
                # Check if this position is already covered
                if all(pos < sp[0] or pos >= sp[1] for sp in special_positions):
                    special_positions.append((pos, pos + len(special_str), token_id))
                start = pos + 1
        
        # Sort special positions by start
        special_positions.sort(key=lambda x: x[0])
        
        # Split text into segments (between special tokens)
        segments = []
        last_end = 0
        for start, end, token_id in special_positions:
            if last_end < start:
                segments.append(("text", text[last_end:start]))
            segments.append(("special", token_id))
            last_end = end
        if last_end < len(text):
            segments.append(("text", text[last_end:]))
        
        # Process segments
        token_ids = []
        for seg_type, content in segments:
            if seg_type == "special":
                token_ids.append(content)
            else:
                # Pre-tokenize
                chunks = re.findall(self.pat, content)
                
                # Process each chunk
                for chunk in chunks:
                    chunk_bytes = chunk.encode("utf-8")
                    word = [bytes([b]) for b in chunk_bytes]
                    
                    # Apply BPE merges
                    word = self._apply_bpe_merges(word)
                    
                    # Look up token IDs
                    for token in word:
                        if token in self.reverse_vocab:
                            token_ids.append(self.reverse_vocab[token])
                        else:
                            # Fallback: if token not found, try to split further
                            # This shouldn't happen if vocab is complete, but handle gracefully
                            for byte in token:
                                byte_token = bytes([byte])
                                if byte_token in self.reverse_vocab:
                                    token_ids.append(self.reverse_vocab[byte_token])
        
        return token_ids
    
    def decode(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        
        # Look up bytes for each token ID
        byte_list = []
        for token_id in token_ids:
            if token_id in self.vocab:
                byte_list.append(self.vocab[token_id])
            else:
                # Handle missing token IDs gracefully
                continue
        
        # Concatenate all bytes
        all_bytes = b''.join(byte_list)
        
        # Decode to UTF-8 string
        try:
            return all_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # Fallback: decode with error handling
            return all_bytes.decode("utf-8", errors="replace")
    
    def encode_iterable(self, file_handle):
        """Encode text from a file handle incrementally."""
        for line in file_handle:
            ids = self.encode(line)
            for token_id in ids:
                yield token_id