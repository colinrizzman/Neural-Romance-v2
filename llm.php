<?php

/*
    github.com/colinrizzman

    -----

    This uses llama.cpp (https://github.com/ggml-org/llama.cpp) to query an LLM model to generate training data.
    
    LLM Model:  Qwen3-30B-A3B-Instruct-2507-Q4_K_M
    Model Card: https://huggingface.co/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-GGUF
    Download:   https://huggingface.co/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-GGUF/blob/main/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf

    Potential improvements:
    - The random numbers for the next prompt could be created while waiting for the current prompt to be generated.

    Prerequisits:
    - apt install php-cli php-curl
    
    -----

    llama-server --list-devices
    llama-server --port 8081 --device Vulkan3 --threads 16 --mlock -m Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf

    --device Vulkan1,Vulkan3
    --device none
    --no-kv-offload
    --no-mmap
    --cpu-strict 1
    --ctx-size 512
    --batch-size 512
    --n-gpu-layers -1
    
*/

if($argc < 2){die("Usage: php q.php <port>\n");}
$port = intval($argv[1]);
$inputs = array();
$num_inputs = 26;

function rndPrompt()
{
    global $inputs, $num_inputs;
    for($i = 0; $i < $num_inputs; $i++){$inputs[$i] = random_int(0, 9);}
    $prompt  = "The scale is:\n";
    $prompt  = "0	Not me at all\n";
    $prompt  = "1	Very untrue of me\n";
    $prompt  = "2	Mostly untrue of me\n";
    $prompt  = "3	Somewhat untrue of me\n";
    $prompt  = "4	Slightly untrue of me\n";
    $prompt  = "5	Slightly true of me\n";
    $prompt  = "6	Somewhat true of me\n";
    $prompt  = "7	Mostly true of me\n";
    $prompt  = "8	Very true of me\n";
    $prompt  = "9	Absolutely me\n\n";
    $prompt  = "these attributes describe the subject:\n\n";
    $prompt .= "obese/fat: " . $inputs[0] . "\n";
    $prompt .= "curious: " . $inputs[1] . "\n";
    $prompt .= "empathetic: " . $inputs[2] . "\n";
    $prompt .= "ambitious: " . $inputs[3] . "\n";
    $prompt .= "depressive: " . $inputs[4] . "\n";
    $prompt .= "creative: " . $inputs[5] . "\n";
    $prompt .= "intellectual: " . $inputs[6] . "\n";
    $prompt .= "spiritual: " . $inputs[7] . "\n";
    $prompt .= "traditional: " . $inputs[8] . "\n";
    $prompt .= "loyal: " . $inputs[9] . "\n";
    $prompt .= "dependable: " . $inputs[10] . "\n";
    $prompt .= "emotional: " . $inputs[11] . "\n";
    $prompt .= "nurturing: " . $inputs[12] . "\n";
    $prompt .= "affectionate: " . $inputs[13] . "\n";
    $prompt .= "possessive: " . $inputs[14] . "\n";
    $prompt .= "dominant: " . $inputs[15] . "\n";
    $prompt .= "open minded: " . $inputs[16] . "\n";
    $prompt .= "defiant: " . $inputs[17] . "\n";
    $prompt .= "independent: " . $inputs[18] . "\n";
    $prompt .= "trustworthy: " . $inputs[19] . "\n";
    $prompt .= "social: " . $inputs[20] . "\n";
    $prompt .= "humorous: " . $inputs[21] . "\n";
    $prompt .= "risk-taking: " . $inputs[22] . "\n";
    $prompt .= "adventurous: " . $inputs[23] . "\n";
    $prompt .= "quirky: " . $inputs[24] . "\n";
    $prompt .= "crazy: " . $inputs[25] . "\n";
    $prompt .= "\n\nBased on these attributes give me a 0-100 number of how likely the subject is to find love, explain, use deep reasoning and cite sources to come to the final number, don't give advice and at the end print the number with preceding =";
    #return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n" . $prompt . "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"; # llama + system prompt
    #return "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n" . $prompt . "<|im_end|>\n<|im_start|>assistant\n"; # qwen + system prompt
    #return "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n" . $prompt . "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"; # llama
    return "<|im_start|>user\n" . $prompt . "<|im_end|>\n<|im_start|>assistant\n"; # qwen
}

/*
    prompt	            Text prompt to complete
    n_predict	        Max tokens to generate
    temperature	        Sampling temperature
    top_p	            Nucleus sampling (probability mass cutoff)
    top_k	            Top-K token sampling
    repeat_penalty	    Penalize recently seen tokens
    presence_penalty	Similar to OpenAI penalties
    seed	            RNG seed (per-request override)
    stop	            Array of strings that stop generation
    n_probs	            If >0, returns top-N token probabilities per step
    grammar	            Grammar-based constraint (structured output)
    cache_prompt	    Whether to reuse KV cache for same prefix
    stream	            Stream output (true for SSE-like streaming)
    penalize_nl	        Whether newlines are penalized in repeat penalty

    tfs_z, typical_p, min_p	                Alternative sampling filters
    mirostat, mirostat_tau, mirostat_eta	Mirostat adaptive sampling (0 = off)
*/
$payload = [
    "prompt" => rndPrompt(),
    "top_k" => 20,
    "top_p" => 0.8,
    "min_p" => 0,
    "cache_prompt" => true,
    "n_predict" => -1,
    "temperature" => 0.7,
    "seed" => random_int(0, PHP_INT_MAX),
];

$ch = curl_init();
$options = [
    CURLOPT_URL => "http://127.0.0.1:" . $port . "/completion",
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_POST => true,
    CURLOPT_HTTPHEADER => ["Content-Type: application/json"],
    CURLOPT_POSTFIELDS => json_encode($payload),
];
curl_setopt_array($ch, $options);
if(!is_dir('responses')){mkdir('responses', 0777, true);}
function reverseCut($str)
{
    $result = "";
    for($i = strlen($str) - 1; $i >= 0; $i--)
    {
        $ch = $str[$i];
        if($ch === ' ' || $ch === '='){break;}
        $result = $ch . $result;
    }
    return $result;
}
function getNextHighestTxtNumber($directory)
{
    $directory = rtrim($directory, DIRECTORY_SEPARATOR) . DIRECTORY_SEPARATOR;
    $files = glob($directory . "*.txt");
    $numbers = [];
    foreach ($files as $file)
    {
        $name = pathinfo($file, PATHINFO_FILENAME);
        if(is_numeric($name)){$numbers[] = (int)$name;}
    }
    return !empty($numbers) ? max($numbers)+1 : 0;
}
while(1)
{
    $payload['prompt'] = rndPrompt();
    $payload['seed']   = random_int(0, PHP_INT_MAX);
    curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
    $response = curl_exec($ch);

    if ($response === false) {
        echo "curl error: " . curl_error($ch) . "\n";
        continue;
    }

    $decoded = json_decode($response);
    if (!isset($decoded->content)) {
        echo "error: no content\n";
        continue;
    }

    $content = $decoded->content;
    $out = reverseCut(str_replace("\n", "", $content));
    if (!is_numeric($out)) {
        echo "failed: " . $out . "\n";
        continue;
    }

    $output = "";
    for ($i = 0; $i < $num_inputs; $i++) {
        $output .= number_format(floatval($inputs[$i]) / 9, 2) . " ";
    }
    $output .= number_format(floatval($out) / 100, 2);

    // -------- Begin critical section with retrying open + lock ----------
    // Open (retry until available)
    $td = false;
    while ($td === false) {
        $td = @fopen('training_data.txt', 'a'); // append; creates if missing
        if ($td === false) {
            usleep(200000); // wait 200ms and try again
        }
    }

    // Try-lock loop (wait until lock is available)
    while (!flock($td, LOCK_EX | LOCK_NB)) {
        usleep(200000); // wait 200ms and try again
    }

    // Perform both writes while holding the training_data.txt lock
    $respPath = "responses/" . strval(getNextHighestTxtNumber('responses')) . ".txt";
    file_put_contents($respPath, $content, FILE_APPEND);
    fwrite($td, $output . "\n");
    fflush($td);

    // Release lock and close
    flock($td, LOCK_UN);
    fclose($td);
    // -------- End critical section ----------
}

curl_close($ch);
