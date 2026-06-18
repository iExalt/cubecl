#[cfg(not(target_family = "wasm"))]
mod _impl {
    use std::thread::JoinHandle;

    #[derive(Debug)]
    pub struct WgpuPoll {
        active_handle: std::sync::Arc<()>,
        cancel_sender: std::sync::mpsc::Sender<()>,
        poll_thread: Option<JoinHandle<()>>,
    }

    impl WgpuPoll {
        pub fn new(device: wgpu::Device) -> Self {
            let active_handle = std::sync::Arc::new(());
            let thread_check = active_handle.clone();

            let (cancel_sender, cancel_receiver) = std::sync::mpsc::channel();
            let poll_thread = std::thread::spawn(move || {
                loop {
                    // Check whether the WgpuPoll, this thread, and something else is holding
                    // a handle.
                    if std::sync::Arc::strong_count(&thread_check) > 2 {
                        if let Err(e) = device.poll(wgpu::PollType::Wait {
                            submission_index: None, // Wait for most recent
                            timeout: None,
                        }) {
                            log::warn!(
                                "wgpu: requested wait timed out before the submission was completed during sync. ({e})"
                            )
                        }
                    } else {
                        // Do not cancel thread while someone still needs to poll.
                        if cancel_receiver.try_recv().is_ok() {
                            break;
                        }

                        std::thread::park();
                    }
                    std::thread::yield_now();
                }
            });

            Self {
                active_handle,
                cancel_sender,
                poll_thread: Some(poll_thread),
            }
        }
        /// Get a handle, as long as it's alive the polling will be active.
        pub fn start_polling(&self) -> std::sync::Arc<()> {
            let handle = self.active_handle.clone();
            if let Some(poll_thread) = &self.poll_thread {
                poll_thread.thread().unpark();
            }
            handle
        }

        /// Stops and joins the polling thread.
        pub fn shutdown(&mut self) {
            let Some(poll_thread) = self.poll_thread.take() else {
                return;
            };

            if self.cancel_sender.send(()).is_err() {
                log::warn!("wgpu polling thread exited before shutdown");
            }
            poll_thread.thread().unpark();
            if poll_thread.join().is_err() {
                log::warn!("wgpu polling thread panicked during shutdown");
            }
        }
    }

    impl Drop for WgpuPoll {
        fn drop(&mut self) {
            self.shutdown();
        }
    }
}

// On Wasm, the browser handles the polling loop, so we don't need anything.
#[cfg(target_family = "wasm")]
mod _impl {
    #[derive(Debug)]
    pub struct WgpuPoll {}
    impl WgpuPoll {
        pub fn new(_device: wgpu::Device) -> Self {
            Self {}
        }
        pub fn start_polling(&self) -> alloc::sync::Arc<()> {
            alloc::sync::Arc::new(())
        }
        pub fn shutdown(&mut self) {}
    }
}

pub(crate) use _impl::*;
