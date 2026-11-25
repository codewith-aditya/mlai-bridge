//go:build windows

package main

// proc_windows.go — Windows has no process groups here; taskkill /T walks the child tree.

import (
	"os/exec"
	"strconv"
)

func setProcGroup(cmd *exec.Cmd) {}

// killTree kills cmd and its descendants.
func killTree(cmd *exec.Cmd) error {
	if cmd.Process == nil {
		return nil
	}
	exec.Command("taskkill", "/T", "/F", "/PID", strconv.Itoa(cmd.Process.Pid)).Run()
	return cmd.Process.Kill()
}
