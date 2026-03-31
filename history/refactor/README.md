# Markdown Refactor Tracker

This directory replaces the old issue workflow for the current reliability and refactor push.

## Files

- `BOARD.md`: human-readable summary of all active refactor tickets
- `WORKLOG.md`: append-only ledger of work as it happens
- `tickets/`: one markdown file per ticket
- `locks/`: one markdown file per active claim

## Ticket Lifecycle

1. Pick a ticket from `BOARD.md`.
2. Claim it by creating `locks/<TICKET-ID>--<agent>.md`.
3. Update the matching ticket file and `BOARD.md` in the same change.
4. Append a short entry to `WORKLOG.md`.
5. When done, delete the lock file and set the ticket status to `done`.

## Lock Rules

- One active lock file per ticket.
- If a ticket already has a lock file, treat it as taken unless the lock explicitly says `released`.
- Lock files are intentionally tiny so multiple agents do not collide in one shared table.

## Status Values

- `todo`
- `in_progress`
- `blocked`
- `done`

## Priority Values

- `P1`: correctness / training-signal integrity
- `P2`: important follow-up or support work
- `P3`: cleanup / lower-risk backlog

## Current Scope

These tickets are derived from:

- `history/2026-03-31-audit-ledger.md`

This tracker is intentionally limited to the refactor and reliability work needed to make training runs trustworthy.
