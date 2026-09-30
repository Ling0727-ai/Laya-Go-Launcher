# Compare the answers two running laya-trt services give for identical requests.
#
#   pwsh -File bench\answer-diff.ps1 -A http://127.0.0.1:8471/api/v1 -B http://127.0.0.1:8472/api/v1
#
# This is the user-visible accuracy check. Comparing raw logits (as
# bench/enginecmp does) shows the numerical difference; comparing the API
# responses shows whether a caller would ever see a different answer. The API
# rounds to four decimals and picks a winning option, so that is the level at
# which a difference matters.
#
# Both services are only read from. Neither is started or stopped here.

param(
    [Parameter(Mandatory = $true)][string]$A,
    [Parameter(Mandatory = $true)][string]$B,
    [string]$LabelA = 'A',
    [string]$LabelB = 'B',
    [string]$Out = ''
)

$ErrorActionPreference = 'Stop'

function New-Questions([int]$n) {
    $q = @{}
    for ($i = 0; $i -lt $n; $i++) {
        switch ($i % 3) {
            0 { $q["q$i"] = @{ type = 'choice'; instructions = 'Which department should handle this request?'
                               criteria = @{ billing = 'invoices, payments, refunds'; technical = 'bugs, outages'; sales = 'pricing' } } }
            1 { $q["q$i"] = @{ type = 'score'; instructions = 'How urgent is this request?'
                               criteria = @('not urgent', 'soon', 'critical deadline') } }
            default { $q["q$i"] = @{ type = 'noul'; instructions = 'Does the user threaten to cancel or leave?' } }
        }
    }
    return $q
}

# A spread of shapes: short and long states, few and many questions, and the
# option counts that select different calibration buckets.
# Eleven options selects the `choice:11+` calibration bucket, which is the one
# the checkpoint ships outside the usable range and both implementations clamp.
$crit11 = @{}
foreach ($i in 1..11) { $crit11["opt$i"] = "description $i" }

$cases = @(
    @{ name = 'short-1q';  state = @{ body = 'We were billed twice for March.' }; q = (New-Questions 1) }
    @{ name = 'short-3q';  state = @{ body = 'We were billed twice for March.' }; q = (New-Questions 3) }
    @{ name = 'short-9q';  state = @{ body = 'We were billed twice for March.' }; q = (New-Questions 9) }
    @{ name = 'medium-3q'; state = @{ body = ('We were billed twice for March. Please refund the duplicate today. ' * 12) }; q = (New-Questions 3) }
    @{ name = 'long-3q';   state = @{ body = ('We were billed twice for March. Please refund the duplicate today. ' * 100) }; q = (New-Questions 3) }
    @{ name = 'json-2q';   state = @{ ticket = @{ subject = 'Payout failing'; messages = @(@{ from = 'customer'; text = 'My Stripe payouts have failed for 3 days.' }) } }; q = (New-Questions 2) }
    @{ name = 'choice11';  state = @{ body = 'Pick one' }
                           q = @{ big = @{ type = 'choice'; instructions = 'Which category?'; criteria = $crit11 } } }
)

function Invoke-Answers {
    param([string]$Base, $State, $Questions)
    $body = @{ state = $State; questions = $Questions } | ConvertTo-Json -Depth 14 -Compress
    return Invoke-RestMethod -Uri "$Base/predict" -Method Post -Body $body `
        -ContentType 'application/json' -TimeoutSec 180
}

function Get-Value {
    param($Answer, [string]$Field)
    $v = $Answer.$Field
    if ($null -eq $v) { return '' }
    return "$v"
}

Write-Host "comparing $LabelA vs $LabelB"
Write-Host "  $A"
Write-Host "  $B"
Write-Host ""

$mismatches = 0
$visible = 0
$rows = @()

foreach ($c in $cases) {
    $ra = Invoke-Answers -Base $A -State $c.state -Questions $c.q
    $rb = Invoke-Answers -Base $B -State $c.state -Questions $c.q

    $ids = @($ra.answers.PSObject.Properties.Name | Sort-Object)
    $idsB = @($rb.answers.PSObject.Properties.Name | Sort-Object)
    if (($ids -join ',') -ne ($idsB -join ',')) {
        Write-Host "FAIL $($c.name): different question ids" -ForegroundColor Red
        $mismatches++
        continue
    }

    $caseDiff = 0
    # $caseVisible counts fields that differ at the precision the API publishes
    # (four decimals), which is what a caller can actually observe.
    $caseVisible = 0
    $maxProbDelta = 0.0
    $details = @()

    foreach ($id in $ids) {
        $aa = $ra.answers.$id
        $ab = $rb.answers.$id

        # The decision itself: the chosen option must be identical.
        if ((Get-Value $aa 'choice') -ne (Get-Value $ab 'choice')) {
            $details += "    $id choice: '$($aa.choice)' vs '$($ab.choice)'"
            $caseDiff++
        }
        if ((Get-Value $aa 'type') -ne (Get-Value $ab 'type')) {
            $details += "    $id type: $($aa.type) vs $($ab.type)"
            $caseDiff++
        }

        # Published numbers are compared at their published precision: a
        # difference that survives rounding to four decimals is visible to a
        # caller, and one that does not is noise.
        foreach ($f in 'score', 'noul', 'confidence') {
            $va = $aa.$f
            $vb = $ab.$f
            if ($null -ne $va -and $null -ne $vb) {
                $ra4 = [math]::Round([double]$va, 4)
                $rb4 = [math]::Round([double]$vb, 4)
                if ($ra4 -ne $rb4) {
                    $details += ("    {0} {1}: {2} vs {3}  (delta {4:N4}, visible at 4dp)" -f `
                        $id, $f, $va, $vb, [math]::Abs([double]$va - [double]$vb))
                    $caseVisible++
                }
            }
        }

        # Probabilities, element by element, at published precision.
        if ($aa.probabilities -and $ab.probabilities) {
            foreach ($k in @($aa.probabilities.PSObject.Properties.Name)) {
                $pa = $aa.probabilities.$k
                $pb = $ab.probabilities.$k
                if ($null -ne $pa -and $null -ne $pb) {
                    $d = [math]::Abs([double]$pa - [double]$pb)
                    if ($d -gt $maxProbDelta) { $maxProbDelta = $d }
                    if ([math]::Round([double]$pa, 4) -ne [math]::Round([double]$pb, 4)) {
                        $details += ("    {0} prob[{1}]: {2} vs {3}  (delta {4:N4}, visible at 4dp)" -f `
                            $id, $k, $pa, $pb, $d)
                        $caseVisible++
                    }
                }
            }
        }

        # act_probability: documented to be unreliable under fp16, so it is
        # reported but never counted as a decision difference.
        if ($null -ne $aa.action -and $null -ne $ab.action) {
            $actA = $aa.action.act_probability
            $actB = $ab.action.act_probability
            if ([double]$actA -ne [double]$actB) {
                $details += ("    {0} act_probability: {1} vs {2}   (auxiliary head)" -f $id, $actA, $actB)
            }
        }
    }

    $status = if ($caseDiff -gt 0) { 'DIFF' } elseif ($caseVisible -gt 0) { 'NUM ' } else { 'ok  ' }
    $color = if ($caseDiff -gt 0) { 'Red' } elseif ($caseVisible -gt 0) { 'Yellow' } else { 'Green' }
    Write-Host ("[{0}] {1,-11} tokens={2,-5} decisions_differ={3} fields_visible={4} max_prob_delta={5:N4}" -f `
        $status, $c.name, $ra.usage.input_tokens, $caseDiff, $caseVisible, $maxProbDelta) -ForegroundColor $color
    foreach ($d in $details) { Write-Host $d -ForegroundColor DarkYellow }

    $mismatches += $caseDiff
    $visible += $caseVisible
    $rows += [ordered]@{
        case            = $c.name
        tokens          = $ra.usage.input_tokens
        decisions_diff  = $caseDiff
        fields_visible  = $caseVisible
        max_prob_delta  = [math]::Round($maxProbDelta, 6)
    }
}

Write-Host ""
if ($mismatches -gt 0) {
    Write-Host "FAIL: $mismatches decision(s) differ between the two engines" -ForegroundColor Red
} elseif ($visible -gt 0) {
    Write-Host "DECISIONS MATCH, NUMBERS DRIFT: $visible published field(s) differ after rounding." -ForegroundColor Yellow
    Write-Host "  Every chosen option is identical, so the decision is unchanged; the confidence," -ForegroundColor Yellow
    Write-Host "  score and probability values move slightly. Treat the delta as the accuracy cost" -ForegroundColor Yellow
    Write-Host "  of the faster precision and decide whether the workload tolerates it." -ForegroundColor Yellow
} else {
    Write-Host "PASS: decisions and every published number match" -ForegroundColor Green
}

if ($Out) {
    $doc = [ordered]@{
        captured_at    = (Get-Date).ToUniversalTime().ToString('o')
        a              = $A
        b              = $B
        label_a        = $LabelA
        label_b        = $LabelB
        mismatches     = $mismatches
        fields_visible = $visible
        cases          = $rows
    }
    $doc | ConvertTo-Json -Depth 6 | Set-Content -Path $Out -Encoding utf8
    Write-Host "wrote $Out"
}

if ($mismatches -gt 0) { exit 1 }
